#!/bin/bash
#SBATCH --job-name=gs_bench_v2
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-9
#SBATCH --output=logs/gs_bench_v2_%a.out
#SBATCH --error=logs/gs_bench_v2_%a.err

mkdir -p logs results/gs_bench_v2

echo "Task $SLURM_ARRAY_TASK_ID starting on $(hostname) at $(date)"

julia --project=. gs_benchmark_v2.jl

echo "Task $SLURM_ARRAY_TASK_ID finished at $(date)"
