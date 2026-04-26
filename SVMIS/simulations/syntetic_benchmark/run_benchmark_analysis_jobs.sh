#!/usr/bin/bash

# slurm directives 
#SBATCH --job-name=svmis_benchmark_analysis
#SBATCH --partition=normal 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=5G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x.%j_%a.out
#SBATCH --error=logs/%x.%j_%a.err 
#SBATCH --mail-user=betti_lorenzo@phd.ceu.edu 
#SBATCH --mail-type=BEGIN,FAIL,END 
#SBATCH --array=0-314

# activate conda and the environment of interest 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate statistical-filters-hoi

# double check the environment
echo "Environment name: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Number of visible CPUs: $SLURM_CPUS_ON_NODE"

# retrieve the ARRAY ID to identify the parameter configuration to run
ARRAY_ID=${SLURM_ARRAY_TASK_ID}

# assign experiment name
EXPERIMENT_NAME="sum_two_numbers_experiment_1"

# define output file
RESULTS_DIR="results"
out_file="${RESULTS_DIR}/experiment_${ARRAY_ID}.json"
echo "Task ID:  ${ARRAY_ID}"
echo "Out dir:  ${out_file}"

# skip if output already exists
if [ -e "$out_file" ]; then
    echo "Skipping config_${task_id} (already done)"
    exit 0
fi

echo ""
echo "Starting script..."
echo ""

# run the job
srun python syntetic_benchmark_analysis.py --params-file configs/benchmark_params_table.json --output-file $out_file --experiment-n $ARRAY_ID
