#!/bin/bash

# List of DATASETs to process
DATASET=droid
CPU=4
RAM=60

# Script to be executed
SCRIPT_PATH="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_track_oxe_subtraj.py"

# Base directory for log files
LOG_DIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/slurm_logs"
mkdir -p "$LOG_DIR"

# Load conda environment
source /lustre/fs12/portfolios/nvr/users/mmemmel/miniforge3/etc/profile.d/conda.sh
conda activate tfds

IMG_KEY=${1:-"primary"} # "primary" "secondary" "tertiary"
TARGET="path_mask"

for ((i = 0; i < 20; i++)); do
    echo
    echo "#######################################################"
    echo

    LOG_FILE="$LOG_DIR/${DATASET}_${IMG_KEY}_${TARGET}_${i}_$(date +%Y%m%d_%H%M%S).log"
    echo "[TRAIN] LOGDIR ${LOG_FILE}"

    # Construct the command
    CMD="python ${SCRIPT_PATH} --task ${DATASET} --save_sketches_every_n 1000 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${IMG_KEY} --target ${TARGET} --train_test_split 1.0"

    echo $CMD
    # Submit the job using srun
    srun -A nvr_srl_simpler \
        -J ${DATASET}_${IMG_KEY}_${TARGET} \
        --cpus-per-task ${CPU} \
        --mem ${RAM}G \
        --partition cpu_long,cpu,cpu_interactive \
        --time 24:00:00 \
        --exclusive \
        --unbuffered \
        bash -c "$CMD" >$LOG_FILE 2>&1

    # Add a short delay between submissions
    sleep 1
done
echo "All jobs submitted. Check status using 'squeue -u \$USER'"