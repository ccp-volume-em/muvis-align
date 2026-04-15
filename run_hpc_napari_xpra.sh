#!/usr/bin/env bash
#SBATCH --job-name=napari_xpra
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=64G   # Memory pool for all cores (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
ml purge
ml Singularity
singularity run --bind /nemo/project/proj-mrc-mm/raw/em/EM04652/EM04652_02_slice017/EM04652-02_slice17_spaghettiandmeatballs2:/data  napari-xpra_latest.sif
