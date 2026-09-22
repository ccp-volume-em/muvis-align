#!/usr/bin/env bash
#SBATCH --job-name=muvis_align
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=640G   # Memory pool for all cores (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
# threads for reading source metadata at project load - almost all of that time is
# spent waiting on a file open, so a shared filesystem wants well above the core count
export MUVIS_SOURCE_INIT_WORKERS=256
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate muvis-align-env
python run.py $1