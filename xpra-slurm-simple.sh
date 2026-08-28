#!/usr/bin/env bash
#SBATCH --job-name=muvis_align
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=640G   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=muvis-align-%j.log
#SBATCH --error=muvis-align-%j.log

export PYTHONUNBUFFERED=TRUE

# ===========================================================================
#  EDIT THESE FOUR LINES
# ===========================================================================
SIF_PATH="/nemo/stp/ddt/working/defoltj/muvis-align/muvis-align-xpra_v0.3.0.sif"          # container on shared storage
DATA_DIR="/nemo/project/proj-ccp-vem/datasets"              # directory to expose
XPRA_PORT=9876                            # port on the compute node

ml Apptainer

apptainer exec "${SIF_PATH}" --bind "${DATA_DIR}:/data" --bind-tcp="0.0.0.0:${XPRA_PORT}"
