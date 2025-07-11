#!/bin/bash

# List of datasets to process
DATASETS=(
    
    # NOT USABLE
    # 'roboturk' # doesn't exist yet
    # 'dobbe' # doesn't exist yet
    # 'language_table' # doesn't have lang
    
    # CONSIDERED USELESS
    # 'furniture_bench_dataset_converted_externally_to_rlds' -> short paths
    # 'berkeley_cable_routing' -> useless task
    # 'nyu_franka_play_dataset_converted_externally_to_rlds' -> same lang instruction: "play with the kitchen"
    # 'kuka' -> short paths, unidentifiable objects
    
    'bridge_v2'
    'droid'
    'dlr_edan_shared_control_converted_externally_to_rlds' 
    'austin_sirius_dataset_converted_externally_to_rlds'
    'toto'
    'taco_play'
    'viola'
    'berkeley_autolab_ur5'
    'stanford_hydra_dataset_converted_externally_to_rlds'
    'austin_buds_dataset_converted_externally_to_rlds'
    'nyu_franka_play_dataset_converted_externally_to_rlds'
    'ucsd_kitchen_dataset_converted_externally_to_rlds'
    'austin_sailor_dataset_converted_externally_to_rlds'
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds'
    'utaustin_mutex'
    'berkeley_fanuc_manipulation'
    'jaco_play'
    'bc_z'
    'cmu_stretch'
    'fmb'
    'fractal20220817_data'
)

TARGET="path_mask_lang" # -> droid doesn't have lang

DEBUG=false

# Script to be executed
SCRIPT_PATH="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_track_oxe_subtraj.py"

# Base directory for log files
LOG_DIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/slurm_logs_oxe_lang"
mkdir -p "$LOG_DIR"

N_SEED=5

# Loop through each target-img_key-dataset and submit a job
for dataset in "${DATASETS[@]}"; do
    if [ "$dataset" == "droid" ]; then
        CPU=8
        RAM=80
    else
        CPU=8
        RAM=30
    fi

    for img_key in "primary" "secondary" "tertiary"; do

        for seed in $(seq 0 $((N_SEED-1))); do

            LOG_FILE="$LOG_DIR/${dataset}_${img_key}_${TARGET}_${seed}_$(date +%Y%m%d_%H%M%S).log"

            echo "Submitting job: $TARGET $img_key $dataset"
            echo "Log file: $LOG_FILE"

            if [ "$dataset" == "bridge_v2" ]; then
                TRAIN_TEST_SPLIT=0.99
                CMD="python ${SCRIPT_PATH} --task ${dataset} --save_sketches_every_n 10 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${img_key} --target ${TARGET} --split train --train_test_split ${TRAIN_TEST_SPLIT} --seed ${seed} && python ${SCRIPT_PATH} --task ${dataset} --save_sketches_every_n 10 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${img_key} --target ${TARGET} --split test --train_test_split ${TRAIN_TEST_SPLIT} --seed ${seed}"
            else
                TRAIN_TEST_SPLIT=1.00
                CMD="python ${SCRIPT_PATH} --task ${dataset} --save_sketches_every_n 10 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${img_key} --target ${TARGET} --train_test_split ${TRAIN_TEST_SPLIT} --seed ${seed}"
            fi
            echo $CMD

            if [ "$DEBUG" = true ]; then
                continue
            fi

            # Submit the job
            sbatch <<EOT
#!/bin/bash
#SBATCH -A nvr_srl_simpler
#SBATCH --partition=cpu_long,cpu
#SBATCH --time=24:00:00
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
${CMD}

# Print completion information
echo "Job completed at: \$(date)"
echo "Exit status: \$?"
EOT
                    
        # Add a short delay between submissions to avoid overwhelming the scheduler
        sleep 1
        done
    done
done

echo "All jobs submitted. Check status using 'squeue -u \$USER'"