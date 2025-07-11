#!/bin/bash

# List of datasets to process
DATASETS=(
    
    'stack_platforms2'
    'libero_90'
    'libero_spatial'
    'libero_goal'
    'libero_object'
    # 'libero_10'
)
CPU=8 # 16
RAM=20 # 20, 176

# Script to be executed
SCRIPT_PATH="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_tracking.py"

# Base directory for log files
LOG_DIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/slurm_logs"
mkdir -p "$LOG_DIR"

# Loop through each target-img_key-dataset and submit a job
for dataset in "${DATASETS[@]}"; do
    for target in "path_mask" "path"; do
        for img_key in "primary" ; do
            for third in "first" "second" "third"; do
                LOG_FILE="$LOG_DIR/${dataset}_${img_key}_${target}_${third}_$(date +%Y%m%d_%H%M%S).log"

                echo "Submitting job: $target $img_key $dataset"
                echo "Log file: $LOG_FILE"

                echo "python ${SCRIPT_PATH} --task ${dataset} --save_sketches_every_n 1000 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${img_key} --target ${target} --train_test_split 1.0 --third ${third}"

                # Submit the job
                sbatch <<EOT
#!/bin/bash
#SBATCH -A nvr_srl_simpler
#SBATCH --partition=cpu_short
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=${CPU}
#SBATCH --mem=${RAM}G
#SBATCH --job-name=${dataset}
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
python ${SCRIPT_PATH} --task ${dataset} --save_sketches_every_n 1000 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${img_key} --target ${target} --train_test_split 1.0 --third ${third}

# Print completion information
echo "Job completed at: \$(date)"
echo "Exit status: \$?"
EOT
                
                # Add a short delay between submissions to avoid overwhelming the scheduler
                sleep 1
            done
        done
    done
done

echo "All jobs submitted. Check status using 'squeue -u \$USER'"