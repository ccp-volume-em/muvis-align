#!/usr/bin/env bash
#SBATCH --job-name=muvis_align
#SBATCH --part=ncpu
#SBATCH --cpus-per-task=64
#SBATCH --time=7-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=64G   # fusion and napari size themselves to it (see also --mem-per-cpu)

export PYTHONUNBUFFERED=TRUE
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
echo "Resources: $(nproc) of $(nproc --all 2>/dev/null || echo '?') CPUs, ${JOB_MEMORY} memory of $(free -g 2>/dev/null | awk '/^Mem:/ { print $2 "GB" }' || echo '?') on $(hostname)"
# threads for reading source metadata at project load - almost all of that time is
# spent waiting on a file open, so a shared filesystem wants well above the core count
export MUVIS_SOURCE_INIT_WORKERS=256
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate muvis-align-env
python run.py $1