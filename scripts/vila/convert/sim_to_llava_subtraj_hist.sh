#!/bin/bash

# List of datasets to process
DATASETS=(
    
    # 'stack_platforms2'
    'libero_90'
    
    'libero_10'
    'libero_spatial'
    'libero_goal'
    'libero_object'
)
CPU=8
RAM=20

DEBUG=false

# Script to be executed
SCRIPT_PATH="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_track_sim_subtraj_hist.py"

# Base directory for log files
LOG_DIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/slurm_logs_sim"
mkdir -p "$LOG_DIR"

target="path_mask_history"
img_key="primary"

N_SEED=5

# Loop through each target-img_key-dataset and submit a job
for dataset in "${DATASETS[@]}"; do

    LOG_FILE="$LOG_DIR/${dataset}_${img_key}_${target}_$(date +%Y%m%d_%H%M%S).log"

    echo "Submitting job: $target $img_key $dataset"
    echo "Log file: $LOG_FILE"

    CMD="python ${SCRIPT_PATH} --task ${dataset} --save_sketches_every_n 1000 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${img_key} --target ${target} --train_test_split 1.0 --num_seeds ${N_SEED}"
    echo $CMD

    if [ "$DEBUG" = true ]; then
        continue
    fi

    # Submit the job
    sbatch <<EOT
#!/bin/bash
#SBATCH -A nvr_srl_simpler
#SBATCH --partition=cpu_short
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=${CPU}
#SBATCH --mem=${RAM}G
#SBATCH --job-name=${dataset}_${seed}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}

# Load conda environment
source /lustre/fs12/portfolios/nvr/users/mmemmel/miniforge3/etc/profile.d/conda.sh
conda activate tfds

# Print basic job information
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"
echo "Dataset: ${dataset}"
echo "Memory allocated: \$(free -h | grep Mem | awk '{print \$2}')"
echo "CPUs allocated: \$(nproc)"

# Run the Python command
${CMD}

# Print completion information
echo "Job completed at: \$(date)"
echo "Exit status: \$?"
EOT
            
# Add a short delay between submissions to avoid overwhelming the scheduler
sleep 1
done
echo "All jobs submitted. Check status using 'squeue -u \$USER'"