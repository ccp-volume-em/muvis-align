#!/usr/bin/env bash
#SBATCH --job-name=muvis_align
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread  # 64 whole cores: given both threads of 32, each worker ran at half speed
#SBATCH --time=2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=64G   # fusion and napari size themselves to it (see also --mem-per-cpu)
#SBATCH --output=muvis-align-%j.log
#SBATCH --error=muvis-align-%j.log

export PYTHONUNBUFFERED=TRUE

# ===========================================================================
#  EDIT THESE LINES
# ===========================================================================
# Refresh with: bash xpra-pull.sh
# Slurm partition: this one where the cluster has it, else the cluster's default (partitions differ per cluster).
# Override with  MUVIS_PARTITION=<name> bash xpra-run.sh  (empty: the cluster's default)
PREFERRED_PARTITION="ncpu"
# the muvis-align folder these scripts are in: a job runs a spool copy, so the launcher below passes it on
if [ -n "${MUVIS_SCRIPT_DIR:-}" ]; then
    SCRIPT_DIR="${MUVIS_SCRIPT_DIR}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    # submitted with sbatch directly: from the folder itself
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
CONTAINER_PATH="${MUVIS_CONTAINER_DIR:-${SCRIPT_DIR}}/muvis-align-xpra_latest"   # built there by xpra-pull.sh
# directory napari can open data from: the first of these the cluster has, else the top-level one the folder is in
# Override with  MUVIS_DATA_DIR=<directory> bash xpra-run.sh
PREFERRED_DATA_DIRS="/nemo"
DATA_DIR="${MUVIS_DATA_DIR:-}"
if [ -z "${DATA_DIR}" ]; then
    for PREFERRED_DATA_DIR in ${PREFERRED_DATA_DIRS}; do
        if [ -z "${DATA_DIR}" ] && [ -d "${PREFERRED_DATA_DIR}" ]; then
            DATA_DIR="${PREFERRED_DATA_DIR}"
        fi
    done
fi
DATA_DIR="${DATA_DIR:-/$(echo "${SCRIPT_DIR}" | cut -d/ -f2)}"
LOGIN_NODE="${MUVIS_LOGIN_NODE:-${SLURM_SUBMIT_HOST:-$(hostname -f)}}"         # what you ssh into
XPRA_PORT=9876                            # first port tried on the compute node

# ---------------------------------------------------------------------------
#  DISPLAY / RESPONSIVENESS TUNING  (optional)
# ---------------------------------------------------------------------------
#  Latency over a tunnel is dominated by video encoding, not by OpenGL.
#  If the GUI feels sluggish, make the browser window smaller, then lower QUALITY.
#
#  RESOLUTION  Largest the virtual screen can grow to. The screen follows the
#              browser tab's size (--resize-display), so a maximised napari
#              fills the tab exactly; a tab larger than this is scaled instead,
#              and then clicks land off target. 8192x4096 costs ~150MB, once.
#
#  ENCODING    auto  - let xpra pick per update (recommended: the image has no
#                      h264 encoder, so h264 is ignored - xpra.log says so)
#              rgb   - lossless, only sensible on a fast LAN
#
#  MIN_QUALITY 1-100. Lower = more compression artefacts, lower latency.
#              The browser client sends its own (10), which xpra info shows
#              overriding this; kept for other clients.
#
#  MIN_SPEED   1-100. Higher = prioritise responsiveness over image quality.
#              70 favours interactivity; 30 favours fidelity.
#
#  Video is off (--video=no below): the browser client caps video at 1024x768,
#  so the napari window was sent at 1/3 size, blurred - idle too, as its jpeg
#  also went through the video path's scaling. The image has no video encoder.
#
#  NOTE: for critical visual assessment of tomograms, raise MIN_QUALITY to
#  90-100 or use ENCODING=rgb, since lossy encoding can mask fine detail.
# ---------------------------------------------------------------------------
RESOLUTION="8192x4096"
ENCODING="auto"
MIN_QUALITY=80
MIN_SPEED=70

# ---------------------------------------------------------------------------
#  SOURCE READING  (optional)
# ---------------------------------------------------------------------------
#  Threads used to read each source's metadata when a project is opened.
#  A thread spends almost all of that waiting on a file open rather than on
#  the CPU - one 34k-tile project measured 364ms per file of which 14ms was
#  CPU - so on a shared filesystem this wants to be well above the core count.
#  Raise it while "Init sources" still reports a low core count; lower it if
#  the filesystem starts complaining.
# ---------------------------------------------------------------------------
SOURCE_INIT_WORKERS=256
#  MALLOC_MMAP_THRESHOLD_ (set on the apptainer line below) has glibc give
#  each decoded tile its own mapping, returned to the OS when freed. Left to
#  glibc's per-thread heaps, one 34k-tile overview kept 240GB after use.
# ===========================================================================

set -euo pipefail

# ===========================================================================
#  USAGE, on the login node:   bash xpra-run.sh
#  Submits this job, waits for it to start and prints how to connect.
#  "sbatch xpra-run.sh" also works: the instructions are then in
#  muvis-align-<job id>.log, and the password in ~/.xpra/job-<job id>/passwd.
#  With MUVIS_FOLLOW=1 (as xpra-connect.sh runs it) it prints one MUVIS_SESSION
#  line instead, with the password, and then waits until the job ends.
# ===========================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
    AVAILABLE_PARTITIONS="$(sinfo -h -o %R 2>/dev/null || true)"
    if [ -n "${MUVIS_PARTITION+set}" ]; then
        PARTITION="${MUVIS_PARTITION}"
    elif grep -qx "${PREFERRED_PARTITION}" <<< "${AVAILABLE_PARTITIONS}"; then
        PARTITION="${PREFERRED_PARTITION}"
    else
        PARTITION=""
    fi
    echo "Partition: ${PARTITION:-default of the cluster}"
    # what this script asks for against the partition's nodes and limits: printed only, never stops the launch
    LIMITS_PARTITION="${PARTITION:-$(sinfo -h -o %P 2>/dev/null | sed -n 's/\*$//p' | head -1 || true)}"
    awk -v cpus="$(sed -n 's/^#SBATCH --cpus-per-task=\([0-9]*\).*/\1/p' "$0" | head -1)" \
        -v mem="$(sed -n 's/^#SBATCH --mem=\([0-9]*[KMGT]\{0,1\}\).*/\1/p' "$0" | head -1)" \
        -v node="$(sinfo -h ${LIMITS_PARTITION:+-p "${LIMITS_PARTITION}"} -o '%c %m' 2>/dev/null | sort -n -k2 | tail -1 || true)" \
        -v limits="$(scontrol show partition ${LIMITS_PARTITION} 2>/dev/null | grep -oE '(DefMemPerCPU|MaxMemPerCPU)=[0-9]+' | tr '\n' ' ' || true)" '
        BEGIN {
            split(node, node_fields, " "); node_cpus = node_fields[1]; node_mb = node_fields[2] + 0
            count = split(limits, limit_fields, " ")
            for (field = 1; field <= count; field++) {
                split(limit_fields[field], pair, "=")
                if (pair[1] == "DefMemPerCPU") default_mb = pair[2] + 0
                if (pair[1] == "MaxMemPerCPU") max_mb = pair[2] + 0
            }
            if (mem != "") {
                unit = substr(mem, length(mem))
                # sbatch reads a bare --mem value as MB
                mem_mb = (mem + 0) * (unit == "T" ? 1048576 : unit == "G" ? 1024 : unit == "K" ? 1 / 1024 : 1)
            }
            printf "Resources: %s CPUs, %s", (cpus == "" ? "?" : cpus),
                   (mem == "" ? "no --mem (the partition default)" : sprintf("%.0f GB", mem_mb / 1024))
            if (cpus != "" && mem != "") printf " (%.1f GB a CPU)", mem_mb / 1024 / cpus
            printf "\n           nodes up to %s CPUs, %s", (node_cpus == "" ? "?" : node_cpus + 0),
                   (node_mb ? sprintf("%.0f GB", node_mb / 1024) : "? GB")
            if (default_mb) printf "; default %.1f GB a CPU", default_mb / 1024
            if (max_mb) printf ", at most %.1f GB a CPU", max_mb / 1024
            printf "\n"
            if (cpus != "" && node_cpus != "" && cpus + 0 > node_cpus + 0)
                print "WARNING: more CPUs than any node of the partition has: the job cannot start"
            if (mem != "" && node_mb && mem_mb > node_mb)
                print "WARNING: more memory than any node of the partition has: the job cannot start"
            if (mem != "" && cpus != "" && max_mb && mem_mb / cpus > max_mb)
                print "WARNING: more memory a CPU than the partition allows: Slurm adds CPUs to match"
            if (mem == "" && cpus != "" && default_mb)
                printf "Note: without --mem the job gets the default, %.0f GB\n", cpus * default_mb / 1024
        }' || true
    export MUVIS_SCRIPT_DIR="${SCRIPT_DIR}"
    JOB_ID="$(sbatch --parsable ${PARTITION:+--partition="${PARTITION}"} "$0")"
    # --parsable adds ";cluster" on a multi-cluster setup
    JOB_ID="${JOB_ID%%;*}"
    LOG_FILE="muvis-align-${JOB_ID}.log"
    PASSWORD_FILE="${HOME}/.xpra/job-${JOB_ID}/passwd"
    trap 'echo; echo "Stopped waiting. Job ${JOB_ID} keeps going - log: ${LOG_FILE}, cancel: scancel ${JOB_ID}"; exit 130' INT
    echo "Submitted job ${JOB_ID}. Waiting for it to start (Ctrl-C stops waiting, not the job)."
    LAST_STATE=""
    # the instructions block is complete once both of its ===== lines are in the log
    while true; do
        BLOCK_LINES="$(grep -c '^=\+$' "${LOG_FILE}" 2>/dev/null || true)"
        if [ "${BLOCK_LINES:-0}" -ge 2 ] && [ -s "${PASSWORD_FILE}" ]; then
            break
        fi
        STATE="$(squeue -h -j "${JOB_ID}" -o '%T %R' 2>/dev/null || true)"
        if [ -z "${STATE}" ]; then
            echo "Job ${JOB_ID} ended before the session started: $(sacct -n -X -j "${JOB_ID}" -o State 2>/dev/null | xargs || true)"
            echo "--- end of ${LOG_FILE}:"
            tail -n 20 "${LOG_FILE}" 2>/dev/null || echo "(no log)"
            exit 1
        fi
        if [ "${STATE}" != "${LAST_STATE}" ]; then
            echo "[$(date +%T)] ${STATE}"
            LAST_STATE="${STATE}"
        fi
        sleep 5
    done
    if [ -z "${MUVIS_FOLLOW:-}" ]; then
        awk '/^=+$/ { lines++ } lines { print } lines == 2 && /^=+$/ { exit }' "${LOG_FILE}" \
            | sed "s|^\(      password:  \).*|\1$(cat "${PASSWORD_FILE}")|"
        exit 0
    fi
    echo "$(grep -m1 '^MUVIS_SESSION ' "${LOG_FILE}") user=${USER} password=$(cat "${PASSWORD_FILE}")"
    # a caller over ssh learns of the job's end from this output ending
    while [ -n "$(squeue -h -j "${JOB_ID}" -o %T 2>/dev/null || true)" ]; do
        sleep 10
    done
    echo "Job ${JOB_ID} ended: $(sacct -n -X -j "${JOB_ID}" -o State 2>/dev/null | xargs || true)"
    exit 0
fi

# Batch jobs do not inherit your interactive module environment.
if ! command -v apptainer >/dev/null 2>&1; then
    module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found. Try: module avail apptainer" >&2
    exit 1
fi
if [ ! -e "${CONTAINER_PATH}" ]; then
    echo "ERROR: container not found at ${CONTAINER_PATH}" >&2
    echo "       Run: bash ${SCRIPT_DIR}/xpra-pull.sh  (or set MUVIS_CONTAINER_DIR to where it is)" >&2
    exit 1
fi

XPRA_HOME="${HOME}/.xpra"
RUN_DIR="${XPRA_HOME}/job-${SLURM_JOB_ID}"
PASSWORD_FILE="${RUN_DIR}/passwd"
# the job folder's path in the container: outside any data folder, whose bind would otherwise hide it
JOB_DIR="/muvis-job"

mkdir -p "${RUN_DIR}/xkb"
chmod 700 "${XPRA_HOME}" "${RUN_DIR}"

# --- one-time password, readable only by you -------------------------------
XPRA_PASS="$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 20)"
printf '%s' "${XPRA_PASS}" > "${PASSWORD_FILE}"
chmod 600 "${PASSWORD_FILE}"

# --- always clean up, however the job ends ---------------------------------
cleanup() {
    rm -f "${PASSWORD_FILE}"
    echo "[$(date)] Session ended; password file removed."
}
trap cleanup EXIT

COMPUTE_NODE="$(hostname -s)"
COMPUTE_NODE_TCP="$(hostname -i)"
LOCAL_PORT=9876
# another job on a shared node may hold the port already
while (exec 3<>"/dev/tcp/127.0.0.1/${XPRA_PORT}") 2>/dev/null; do
    XPRA_PORT=$((XPRA_PORT + 1))
done

# the node's hardware and load, so a slow run can be put down to the node it ran on
lscpu_value() { lscpu 2>/dev/null | awk -F: -v key="$1" '$1 == key { gsub(/^ +/, "", $2); print $2; exit }'; }
CPU_MODEL="$(lscpu_value 'Model name')"
CPU_LAYOUT="$(lscpu_value 'Socket(s)') sockets x $(lscpu_value 'Core(s) per socket') cores x $(lscpu_value 'Thread(s) per core') threads"
CPU_ALLOCATED="$(nproc) of $(lscpu_value 'CPU(s)') CPUs allocated (${SLURM_CPUS_ON_NODE:-?} by Slurm)"
# fewer physical cores than CPUs: workers share cores as hyperthreads, each pair taking up to twice the CPU time
CPU_LIST="$(awk '/^Cpus_allowed_list/ { print $2 }' /proc/self/status)"
# "<cores> <first CPU of each core, comma-separated>"
CORE_CPUS="$(lscpu -p=CPU,CORE 2>/dev/null | awk -F, -v list="${CPU_LIST}" '
    BEGIN { count = split(list, ranges, ",")
            for (index_ = 1; index_ <= count; index_++) {
                split(ranges[index_], bounds, "-"); last = (bounds[2] == "") ? bounds[1] : bounds[2]
                for (cpu = bounds[1] + 0; cpu <= last + 0; cpu++) allowed[cpu] = 1 } }
    /^#/ { next }
    ($1 + 0) in allowed && !($2 in cores) { cores[$2] = 1; total++; firsts = firsts (firsts == "" ? "" : ",") $1 }
    END { print total + 0, firsts }')"
PHYSICAL_CORES="${CORE_CPUS%% *}"
# the plugin sizes its worker pools by the CPUs it may run on: one thread of each core gives a worker a core
PINNED_CPUS="${CORE_CPUS#* }"
PIN_TO_CORES=()
PIN_NOTE=""
if [ -n "${PINNED_CPUS}" ] && [ "${PHYSICAL_CORES}" -lt "$(nproc)" ] && command -v taskset >/dev/null 2>&1; then
    PIN_TO_CORES=(taskset -c "${PINNED_CPUS}")
    PIN_NOTE="
            napari pinned to one thread of each core"
fi
NODE_LOAD="$(cut -d' ' -f1-3 /proc/loadavg) (1/5/15 min), $(free -g 2>/dev/null | awk '/^Mem:/ { print $7 "GB of " $2 "GB free" }')"
# the job's memory as the plugin reads it: Slurm's variables, else the tightest limit on its cgroup or a parent
JOB_MEMORY="$({ cat /proc/self/cgroup 2>/dev/null || true; } | awk -F: -v node_mb="${SLURM_MEM_PER_NODE:-}" -v cpu_mb="${SLURM_MEM_PER_CPU:-}" -v cpus="$(nproc)" '
    {
        root = ($2 == "") ? "/sys/fs/cgroup" : ($2 ~ /(^|,)memory(,|$)/ ? "/sys/fs/cgroup/memory" : "")
        name = ($2 == "") ? "memory.max" : "memory.limit_in_bytes"
        count = split($3, parts, "/")
        path = root
        for (depth = 1; root != "" && depth <= count + 1; depth++) {
            file = path "/" name
            if ((getline value < file) > 0 && value ~ /^[0-9]+$/ && value + 0 < 2 ^ 60 && (lowest == "" || value + 0 < lowest))
                lowest = value + 0
            close(file)
            if (depth <= count && parts[depth] != "") path = path "/" parts[depth]
        }
    }
    END {
        if (node_mb != "") printf "%.0fGB (Slurm)", node_mb / 1024
        else if (cpu_mb != "") printf "%.0fGB (Slurm)", cpu_mb * cpus / 1024
        else if (lowest != "") printf "%.0fGB (cgroup)", lowest / 1073741824
        else printf "? (no limit found)"
    }' || echo '?')"

echo "MUVIS_SESSION job=${SLURM_JOB_ID} node=${COMPUTE_NODE} port=${XPRA_PORT}"
cat <<EOF

======================================================================
  muvis-align / XPRA SESSION READY
  Job ID  : ${SLURM_JOB_ID}
  Node    : ${COMPUTE_NODE} ${COMPUTE_NODE_TCP}
  CPU     : ${CPU_MODEL}
            ${CPU_LAYOUT}, ${CPU_ALLOCATED}
            on ${PHYSICAL_CORES} physical cores (CPUs ${CPU_LIST})${PIN_NOTE}
  Memory  : ${JOB_MEMORY} allocated, of $(free -g 2>/dev/null | awk '/^Mem:/ { print $2 "GB" }' || echo '?') on the node
  Load    : ${NODE_LOAD}
  Started : $(date)
----------------------------------------------------------------------

  STEP 1 - On your local machine, open a new terminal and run:

      ssh -N -L ${LOCAL_PORT}:${COMPUTE_NODE}:${XPRA_PORT} ${USER}@${LOGIN_NODE}

    If your local ssh config has an alias for the compute node, with credentials:

      ssh -N -L ${LOCAL_PORT}:${COMPUTE_NODE_TCP}:${XPRA_PORT} ${COMPUTE_NODE}

    Leave this terminal open for the whole session.

  STEP 2 - Open this address in your browser:

      http://localhost:${LOCAL_PORT}/

  STEP 3 - Log in with:

      username:  ${USER}
      password:  run: cat ${PASSWORD_FILE}

  STEP 4 - To finish:

      Close the napari window inside the browser which should automatically disconnect the session, then close the browser tab

      In case napari refuses to close, optionally manually kill the HPC task: scancel ${SLURM_JOB_ID}

  In case something goes wrong, the xpra server log is at:
      ${RUN_DIR}/xpra.log

======================================================================

EOF


# napari with the plugin open, maximised: to the browser tab's size, as the screen follows it
XPRA_START="python3 -c \"import napari; viewer = napari.Viewer(); viewer.window.add_plugin_dock_widget('muvis-align'); viewer.window._qt_window.showMaximized(); napari.run()\""

# Xvfb writes its keymap to /var/lib/xkb when access() calls it writable (the user owns the sandbox's files),
# but the image is read-only there: the keymap fails and Xvfb exits, so give it a writable directory.
# Seamless mode exposes napari as an individual window.  A screen of fixed size
# scaled into the browser tab put the pointer off target when maximised (Chrome 154).
${PIN_TO_CORES[@]+"${PIN_TO_CORES[@]}"} apptainer exec \
    --cleanenv \
    --containall \
    --home "${RUN_DIR}:${JOB_DIR}" \
    --pwd "${JOB_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${RUN_DIR}/xkb:/var/lib/xkb" \
    --env "USER=${USER}" \
    --env "XDG_RUNTIME_DIR=${JOB_DIR}" \
    --env "MUVIS_SOURCE_INIT_WORKERS=${SOURCE_INIT_WORKERS}" \
    ${SLURM_MEM_PER_NODE:+--env "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE}"} \
    ${SLURM_MEM_PER_CPU:+--env "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU}"} \
    --env "MALLOC_MMAP_THRESHOLD_=1048576" \
    "${CONTAINER_PATH}" \
    xpra start \
        --bind-tcp="0.0.0.0:${XPRA_PORT},auth=file:filename=${JOB_DIR}/passwd" \
        --html=on \
        --daemon=no \
        --exit-with-children=yes \
        --start-child="$XPRA_START" \
        --socket-dir="${JOB_DIR}" \
        --log-dir="${JOB_DIR}" \
        --xvfb="Xvfb +extension GLX +extension Composite +extension RANDR -screen 0 ${RESOLUTION}x24 -nolisten tcp -noreset" \
        --resize-display=yes \
        --encoding="${ENCODING}" \
        --min-quality="${MIN_QUALITY}" \
        --min-speed="${MIN_SPEED}" \
        --video-scaling=0 \
        --video=no \
        --dpi=96 \
        --sharing=no \
        --file-transfer=off \
        --printing=no \
        --webcam=no \
        --pulseaudio=no \
        --notifications=no \
        --speaker=off \
        --microphone=off \
        --mdns=no \
        --dbus-control=no \
        --title="muvis-align @ ${COMPUTE_NODE}" \
    2>&1 | tee "${RUN_DIR}/xpra.log"

# Output also goes to ${RUN_DIR}/xpra.log for convenience, in addition to
# the Slurm log.
