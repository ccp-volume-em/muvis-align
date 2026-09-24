#!/usr/bin/env bash
#SBATCH --job-name=muvis_align
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=640G   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=muvis-align-%j.log
#SBATCH --error=muvis-align-%j.log

export PYTHONUNBUFFERED=TRUE

# ===========================================================================
#  EDIT THESE FOUR LINES
# ===========================================================================
# Refresh with: sbatch xpra-pull.sh
CONTAINER_PATH="/nemo/stp/ddt/working/defoltj/muvis-align/muvis-align-xpra_latest"          # container on shared storage
DATA_DIR="/nemo/project"              # directory to expose
LOGIN_NODE="login.nemo.thecrick.org"         # what you ssh into
XPRA_PORT=9876                            # port on the compute node

# ---------------------------------------------------------------------------
#  DISPLAY / RESPONSIVENESS TUNING  (optional)
# ---------------------------------------------------------------------------
#  Latency over a tunnel is dominated by video encoding, not by OpenGL.
#  If the GUI feels sluggish, lower RESOLUTION first, then QUALITY.
#
#  RESOLUTION  Maximum virtual screen size available to application windows.
#              In seamless mode the browser controls each window's actual size.
#              Smaller = fewer pixels to encode = snappier.
#              1920x1080 (default) | 1600x900 | 1280x1024 | 1280x720
#
#  ENCODING    h264  - best latency/bandwidth for GUI work (recommended)
#              vp9   - better compression, more CPU
#              rgb   - lossless, only sensible on a fast LAN
#              auto  - let xpra decide
#
#  MIN_QUALITY 1-100. Lower = more compression artefacts, lower latency.
#              50 is a good balance; use 80+ if judging image detail.
#
#  MIN_SPEED   1-100. Higher = prioritise responsiveness over image quality.
#              70 favours interactivity; 30 favours fidelity.
#
#  NOTE: for critical visual assessment of tomograms, raise MIN_QUALITY to
#  90-100 or use ENCODING=rgb, since lossy encoding can mask fine detail.
# ---------------------------------------------------------------------------
RESOLUTION="1920x1080"
ENCODING="h264"
MIN_QUALITY=50
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
#  1 logs what holds large buffers in memory during and after the view's
#  overview (seconds per check on a large project) - for diagnosing only.
LOG_LIVE_BUFFERS=1
#  MALLOC_MMAP_THRESHOLD_ (set on the apptainer line below) has glibc give
#  each decoded tile its own mapping, returned to the OS when freed. Left to
#  glibc's per-thread heaps, one 34k-tile overview kept 240GB after use.
# ===========================================================================

set -euo pipefail

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
    echo "       Run: sbatch xpra-pull.sh" >&2
    exit 1
fi

XPRA_HOME="${HOME}/.xpra"
RUN_DIR="${XPRA_HOME}/job-${SLURM_JOB_ID}"
PASSWORD_FILE="${RUN_DIR}/passwd"

mkdir -p "${RUN_DIR}"
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

cat <<EOF

======================================================================
  muvis-align / XPRA SESSION READY
  Job ID  : ${SLURM_JOB_ID}
  Node    : ${COMPUTE_NODE} ${COMPUTE_NODE_TCP}
  Started : $(date)
----------------------------------------------------------------------

  STEP 1 - On your local machine, open a new terminal and run:

      ssh -N -L ${LOCAL_PORT}:${COMPUTE_NODE}:${XPRA_PORT} ${USER}@${LOGIN_NODE}

    In case locally configured ssh Crick aliases with credential:

      ssh -N -L ${LOCAL_PORT}:${COMPUTE_NODE_TCP}:${XPRA_PORT} ${COMPUTE_NODE}

    Leave this terminal open for the whole session.

  STEP 2 - Open this address in your browser:

      http://localhost:${LOCAL_PORT}/

  STEP 3 - Log in with:

      username:  ${USER}
      password:  ${XPRA_PASS}

  STEP 4 - To finish: close the browser tab, press Ctrl+C in the ssh
    terminal, then run:

      scancel ${SLURM_JOB_ID}

  If something goes wrong, the xpra server log is at:
      ${RUN_DIR}/xpra.log

======================================================================

EOF


XPRA_START="python3 -m napari --with muvis-align"

# Seamless mode exposes napari as an individual window.  In desktop mode
# napari maximizes against the fixed-size Xvfb desktop, which can be larger
# than the browser canvas and leaves controls outside the visible area.
apptainer exec \
    --cleanenv \
    --containall \
    --home "${RUN_DIR}" \
    --pwd "${RUN_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --env "USER=${USER}" \
    --env "XDG_RUNTIME_DIR=${RUN_DIR}" \
    --env "MUVIS_SOURCE_INIT_WORKERS=${SOURCE_INIT_WORKERS}" \
    --env "MUVIS_LOG_LIVE_BUFFERS=${LOG_LIVE_BUFFERS}" \
    --env "MALLOC_MMAP_THRESHOLD_=1048576" \
    "${CONTAINER_PATH}" \
    xpra start \
        --bind-tcp="0.0.0.0:${XPRA_PORT},auth=file:filename=${PASSWORD_FILE}" \
        --html=on \
        --daemon=no \
        --exit-with-children=yes \
        --start-child="$XPRA_START" \
        --socket-dir="${RUN_DIR}" \
        --log-dir="${RUN_DIR}" \
        --xvfb="Xvfb +extension GLX +extension Composite -screen 0 ${RESOLUTION}x24 -nolisten tcp -noreset" \
        --encoding="${ENCODING}" \
        --min-quality="${MIN_QUALITY}" \
        --min-speed="${MIN_SPEED}" \
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
