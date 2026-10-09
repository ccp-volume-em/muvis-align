#!/usr/bin/env bash
#SBATCH --job-name=muvis_align_pull
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=muvis-align-pull-%j.log
#SBATCH --error=muvis-align-pull-%j.log
#
# Refreshes the container used by xpra-slurm.sh: pulls "latest" from quay.io
# and builds it into a fixed, version-less sandbox directory.
#
# Submit with: sbatch xpra-pull.sh
# Needs internet access to quay.io; if compute nodes lack it, run the
# apptainer build line below on the login node instead.

set -euo pipefail

IMAGE_REF="docker://quay.io/ccp-volume-em/muvis-align-xpra:latest"
DEST_DIR="/nemo/stp/ddt/working/defoltj/muvis-align"
SANDBOX_PATH="${DEST_DIR}/muvis-align-xpra_latest"

if ! command -v apptainer >/dev/null 2>&1; then
    module load Apptainer 2>/dev/null || module load apptainer 2>/dev/null || true
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not found. Try: module avail apptainer" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

echo "Pulling ${IMAGE_REF} and building sandbox at ${SANDBOX_PATH} ..."
apptainer build --sandbox --force "${SANDBOX_PATH}" "${IMAGE_REF}"

echo "Done: ${SANDBOX_PATH}"
