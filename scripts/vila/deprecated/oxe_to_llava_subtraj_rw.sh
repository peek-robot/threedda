#!/bin/bash

# List of DATASETs to process
DATASET=bridge_v2
CPU=16
RAM=100


# Script to be executed
SCRIPT_PATH="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/convert_track_oxe_subtraj.py"

# Base directory for log files
LOG_DIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_convert/slurm_logs"
mkdir -p "$LOG_DIR"

# Load conda environment
source /lustre/fs12/portfolios/nvr/users/mmemmel/miniforge3/etc/profile.d/conda.sh
conda activate tfds

IMG_KEY=${1:-"primary"} # "primary" "secondary" "tertiary"

TARGET="path_mask" # "path_mask_lang" "path_mask_lang_rw"

for ((i = 0; i < 20; i++)); do
    echo
    echo "#######################################################"
    echo

    LOG_FILE="$LOG_DIR/${DATASET}_${IMG_KEY}_${TARGET}_${i}_$(date +%Y%m%d_%H%M%S).log"
    echo "[TRAIN] LOGDIR ${LOG_FILE}"

    # Construct the command
    CMD="python ${SCRIPT_PATH} --task ${DATASET} --save_sketches_every_n 1000 --path_rdp_tolerance 0.05 --mask_rdp_tolerance 0.1 --img_key ${IMG_KEY} --target ${TARGET} --train_test_split 1.0 --reword_quest"

    # echo $CMD
    # Submit the job using srun
    srun -A nvr_srl_simpler \
        -J ${DATASET}_${IMG_KEY}_${TARGET} \
        --cpus-per-gpu ${CPU} \
        --gpus 1 \
        --mem ${RAM}G \
        --partition polar,polar3,polar4,grizzly,interactive,interactive_singlenode \
        --time 04:00:00 \
        --exclusive \
        --unbuffered \
        bash -c "$CMD" >$LOG_FILE 2>&1

    # Add a short delay between submissions
    sleep 1
done
echo "All jobs submitted. Check status using 'squeue -u \$USER'"