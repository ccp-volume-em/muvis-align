#!/usr/bin/env bash
#SBATCH --job-name=muvis_align_apptainer_check
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --output=muvis-align-apptainer-check-%j.log
#SBATCH --error=muvis-align-apptainer-check-%j.log
#
# Checks whether a compute node can run the muvis-align-xpra SIF image directly,
# or whether Apptainer unpacks it into a temporary sandbox on every run - in
# which case the permanent sandbox xpra-pull.sh builds is the right choice.
#
# Submit with: sbatch apptainer-check.sh
# Pulls the image (needs internet access to quay.io) as a .sif next to the
# sandbox; the sandbox itself is left as it is. Read the VERDICT at the end.

set -uo pipefail

IMAGE_REF="docker://quay.io/ccp-volume-em/muvis-align-xpra:latest"
DEST_DIR="/nemo/stp/ddt/working/defoltj/muvis-align"
SIF_PATH="${DEST_DIR}/muvis-align-xpra_latest.sif"
SANDBOX_PATH="${DEST_DIR}/muvis-align-xpra_latest"
DEBUG_LOG="${DEST_DIR}/apptainer-check-debug-${SLURM_JOB_ID:-local}.log"

if ! command -v apptainer >/dev/null 2>&1; then
    module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found. Try: module avail apptainer" >&2
    exit 1
fi
mkdir -p "${DEST_DIR}"

echo "=== Node: $(hostname), $(date)"
echo "=== Apptainer: $(apptainer --version)"
echo "user namespaces allowed: $(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo unknown)"
if [ -e /dev/fuse ]; then echo "/dev/fuse: present"; else echo "/dev/fuse: MISSING"; fi
for tool in squashfuse squashfuse_ll fuse2fs fusermount3 fusermount; do
    echo "${tool}: $(command -v "${tool}" || echo 'not found')"
done
echo "APPTAINER_TMPDIR: ${APPTAINER_TMPDIR:-unset (temporary sandboxes go to /tmp)}"
df -h "${APPTAINER_TMPDIR:-/tmp}" | tail -1 | awk '{print "temporary space free: " $4}'

echo
echo "=== Pulling ${IMAGE_REF} to ${SIF_PATH}"
if ! apptainer pull --force "${SIF_PATH}" "${IMAGE_REF}"; then
    echo "VERDICT: could not pull the image here - run the pull on the login node, then submit this again."
    exit 1
fi
ls -lh "${SIF_PATH}"

echo
echo "=== Running the SIF directly (debug log: ${DEBUG_LOG})"
apptainer --debug exec "${SIF_PATH}" true > "${DEBUG_LOG}" 2>&1
SIF_STATUS=$?
echo "exit status: ${SIF_STATUS}"
grep -iE 'squashfuse|fuse|sandbox|temporary|extract|FATAL|WARNING' "${DEBUG_LOG}" | head -20

echo
echo "=== Start-up time of napari + muvis-align"
echo "--- from the SIF"
( time apptainer exec "${SIF_PATH}" python3 -c "import napari, muvis_align" ) 2>&1 | tail -4
if [ -d "${SANDBOX_PATH}" ]; then
    echo "--- from the sandbox"
    ( time apptainer exec "${SANDBOX_PATH}" python3 -c "import napari, muvis_align" ) 2>&1 | tail -4
else
    echo "--- no sandbox at ${SANDBOX_PATH} to compare with"
fi

echo
if [ "${SIF_STATUS}" -ne 0 ]; then
    echo "VERDICT: the SIF does not run here (see FATAL above and ${DEBUG_LOG}) - keep the sandbox."
elif grep -qiE 'converting sif file to temporary sandbox|temporary sandbox' "${DEBUG_LOG}"; then
    echo "VERDICT: the SIF is unpacked into a temporary sandbox on every run - keep building the permanent"
    echo "         sandbox (xpra-pull.sh), or ask for squashfuse on the compute nodes."
elif grep -qi 'squashfuse' "${DEBUG_LOG}"; then
    echo "VERDICT: the SIF is mounted with squashfuse and runs directly - the sandbox is not needed;"
    echo "         compare the start-up times above to choose."
else
    echo "VERDICT: the SIF runs, but the debug log does not say how it was mounted - check ${DEBUG_LOG}."
fi
