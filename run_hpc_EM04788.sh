#!/usr/bin/env bash
#SBATCH --job-name=muvis_align
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=64
#SBATCH --time=240          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=200G  # a large project's fusion runs out at 64G (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate muvis-align-env
python run.py --params=resources/params_EM04788_02_Sub_02_10nm.yml
python run.py --params=resources/params_EM04788_02_Sub_02_20nm.yml
python run.py --params=resources/params_EM04788_03_Sub_01_10nm.yml
python run.py --params=resources/params_EM04788_03_Sub_01_20nm.yml