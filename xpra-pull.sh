#!/usr/bin/env bash
#SBATCH --job-name=muvis_align_pull
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=muvis-align-pull-%j.log
#SBATCH --error=muvis-align-pull-%j.log
#
# Refreshes the container used by xpra-slurm.sh: pulls "latest" from quay.io
# and builds it into a fixed, version-less sandbox directory.
#
# Usage, on the login node:   bash xpra-pull.sh
#   Submits this job and shows its output here until it ends.
#   "sbatch xpra-pull.sh" also works: the output is then in muvis-align-pull-<job id>.log.
# Needs internet access to quay.io; if compute nodes lack it, run the
# apptainer build line below on the login node instead.

set -euo pipefail

IMAGE_REF="docker://quay.io/ccp-volume-em/muvis-align-xpra:latest"
# Slurm partition: this one where the cluster has it, else the cluster's default.
# Override with  MUVIS_PARTITION=<name> bash xpra-pull.sh  (empty: the cluster's default)
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
# where xpra-slurm.sh looks for it
DEST_DIR="${MUVIS_CONTAINER_DIR:-${SCRIPT_DIR}}"
SANDBOX_PATH="${DEST_DIR}/muvis-align-xpra_latest"

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
    export MUVIS_SCRIPT_DIR="${SCRIPT_DIR}"
    JOB_ID="$(sbatch --parsable ${PARTITION:+--partition="${PARTITION}"} "$0")"
    # --parsable adds ";cluster" on a multi-cluster setup
    JOB_ID="${JOB_ID%%;*}"
    LOG_FILE="muvis-align-pull-${JOB_ID}.log"
    TAIL_PID=""
    trap 'echo; echo "Stopped watching. Job ${JOB_ID} keeps going - log: ${LOG_FILE}, cancel: scancel ${JOB_ID}"; [ -n "${TAIL_PID}" ] && kill "${TAIL_PID}" 2>/dev/null; exit 130' INT
    echo "Submitted job ${JOB_ID}. Its output follows (Ctrl-C stops watching, not the job)."
    LAST_STATE=""
    while true; do
        STATE="$(squeue -h -j "${JOB_ID}" -o '%T %R' 2>/dev/null || true)"
        if [ -z "${STATE}" ]; then
            break
        fi
        if [ "${STATE}" != "${LAST_STATE}" ]; then
            echo "[$(date +%T)] ${STATE}"
            LAST_STATE="${STATE}"
        fi
        if [ -z "${TAIL_PID}" ] && [ -e "${LOG_FILE}" ]; then
            tail -n +1 -F "${LOG_FILE}" 2>/dev/null &
            TAIL_PID=$!
        fi
        sleep 5
    done
    # the last lines reach the shared filesystem after the job has left the queue
    sleep 5
    if [ -n "${TAIL_PID}" ]; then
        kill "${TAIL_PID}" 2>/dev/null || true
    else
        cat "${LOG_FILE}" 2>/dev/null || true
    fi
    FINAL_STATE="$(sacct -n -X -j "${JOB_ID}" -o State 2>/dev/null | xargs || true)"
    echo "Job ${JOB_ID} ended: ${FINAL_STATE:-unknown} (log: ${LOG_FILE})"
    if [ "${FINAL_STATE%% *}" = "COMPLETED" ]; then
        exit 0
    fi
    exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
    module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found. Try: module avail apptainer" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

# a fresh sandbox: --force does not clear what an interrupted or older build left behind
echo "Removing ${SANDBOX_PATH} ..."
rm -rf "${SANDBOX_PATH}"

echo "Pulling ${IMAGE_REF} and building sandbox at ${SANDBOX_PATH} ..."
apptainer build --sandbox --force "${SANDBOX_PATH}" "${IMAGE_REF}"

echo "Done: ${SANDBOX_PATH}"
