#!/bin/bash

PRETRAIN=${1:-Efficient-Large-Model/VILA1.5-3b} # VILA1.5-3b vila1.5-v3-13b
DATASET=${2:-rlbench10k_train10k_keypoint_2d_table}
EPOCH=${3:-1}
# LoRA default lr is 2e-4
LR=${4:-2e-4}
OUT_DIR=${5:-/lustre/fsw/portfolios/nvr/users/mmemmel/projects/checkpoints/finetuned/vila/$PRETRAIN-$DATASET-e$EPOCH-LR$LR}
echo "OUT_DIR="$OUT_DIR
mkdir -p $OUT_DIR

SCRIPT='4_lora.sh'

# loging wanbd & activate conda env & run script
CMD="source /lustre/fsw/portfolios/nvr/users/mmemmel/miniforge3/bin/activate vila && \
     export WANDB_API_KEY=638b3ee4d807a3fb5d92aca711d6288f5c3a4aeb && \
     wandb login && \
     cd /lustre/fsw/portfolios/nvr/users/mmemmel/projects/vila/VILA && \
     bash /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts/$SCRIPT \
    $PRETRAIN $DATASET $EPOCH $LR $OUT_DIR"


export NCCL_ASYNC_ERROR_HANDLING=1

for (( i=0; i<1; i++ ))
do
    echo
    echo "#######################################################"
    echo

    # Check if training is skipped bc model exists
    if grep -q "Skipp training" $OUT_DIR/terminal.log; then
        echo "Done"
        break
    fi
    
    srun -A nvr_srl_simpler \
         -J HAMSTER \
         --cpus-per-gpu 16 \
         --gpus 8 \
         --partition polar,polar3,polar4,grizzly,interactive,interactive_singlenode \
         --time 04:00:00 \
         --exclusive \
         --unbuffered \
         bash -c "$CMD" 2>&1 | tee -a $OUT_DIR/terminal.log
    sleep 1m

done

echo "Done."
