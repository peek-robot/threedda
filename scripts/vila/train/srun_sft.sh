#!/bin/bash

PRETRAIN=${1:-Efficient-Large-Model/VILA1.5-3b} # VILA1.5-3b vila1.5-v3-13b
DATASET=${2:-rlbench10k_train10k_keypoint_2d_table}
EPOCH=${3:-1}
LR=${4:-1e-4}
OUT_DIR=${5:-/lustre/fsw/portfolios/nvr/users/mmemmel/projects/checkpoints/finetuned/vila/$PRETRAIN-$DATASET-e$EPOCH-LR$LR}
N_GPUS=${6:-8}

echo "OUT_DIR="$OUT_DIR
mkdir -p $OUT_DIR

# if NVILA in PRETRAIN, use 4_sft_nvila.sh, otherwise use 4_sft_vila.sh
if [[ "$PRETRAIN" == *"NVILA"* ]]; then
    SCRIPT='4_sft_nvila.sh'
    DIR='NVILA'
    CONDA_ENV='nvila'
else
    SCRIPT='4_sft_vila.sh'
    DIR='VILA'
    CONDA_ENV='vila'
fi

# loging wanbd & activate conda env & run script
CMD="source /lustre/fsw/portfolios/nvr/users/mmemmel/miniforge3/bin/activate $CONDA_ENV && \
     export WANDB_API_KEY=638b3ee4d807a3fb5d92aca711d6288f5c3a4aeb && \
     wandb login && \
     cd /lustre/fsw/portfolios/nvr/users/mmemmel/projects/vila/$DIR && \
     bash /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts/$SCRIPT \
    $PRETRAIN $DATASET $EPOCH $LR $OUT_DIR"
echo $CMD

export NCCL_ASYNC_ERROR_HANDLING=1

for (( i=0; i<24; i++ ))
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
         --gpus $N_GPUS \
         --partition polar,polar3,polar4,grizzly,interactive,interactive_singlenode \
         --time 04:00:00 \
         --exclusive \
         --unbuffered \
         bash -c "$CMD" 2>&1 | tee -a $OUT_DIR/terminal.log
    sleep 1m

done

echo "Done."
