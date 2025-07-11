#!/bin/bash

# # e.g., VILA1.5-3b-rlbench_all_tasks_256_1000_eps_sketch_v5_beta-e1-LR1e-5/checkpoint-300
# PRETRAIN=${1:-VILA1.5-3b} # VILA1.5-3b vila1.5-v3-13b
# DATASET=${2:-rlbench10k_train10k_keypoint_2d_table}
# EVALSET=${3:-colosseum_test_375_256_sketch_v5_eval}
# EPOCH=${4:-1}
# LR=${5:-1e-4}
# MODEL_NAME=${6:-$PRETRAIN} # ${6:-$PRETRAIN-$DATASET-e$EPOCH-LR$LR}
# echo "MODEL_NAME="$MODEL_NAME

MODEL_NAME=${1}
EVAL_SETS=${2}

echo $MODEL_NAME
echo $EVAL_SETS

SCRIPT='/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts/4_eval.sh'

IFS='+' read -r -a EVAL_SETS_ARRAY <<< "$EVAL_SETS"  # Split EVAL_SETS by '+'
for EVALSET in "${EVAL_SETS_ARRAY[@]}"; do
    
    CMD="cd /lustre/fsw/portfolios/nvr/users/mmemmel/projects/vila/VILA && \
        source /lustre/fsw/portfolios/nvr/users/mmemmel/miniforge3/bin/activate vila && \
        bash $SCRIPT \
        $MODEL_NAME $EVALSET"
    #     $MODEL_NAME $EVALSET 0"
        # remove 0 above for multi GPU

    export NCCL_ASYNC_ERROR_HANDLING=1

    srun -A nvr_srl_simpler \
            -J HAMSTER \
            --cpus-per-gpu 16 \
            --gpus 8 \
            --partition polar,polar3,polar4,grizzly,interactive,interactive_singlenode\
            --time 04:00:00 \
            --exclusive \
            --unbuffered \
            bash -c "$CMD" 2>&1 
    echo "Evaluated $MODEL_NAME on $EVALSET"
            # set gpus back to 8
done

# aggregate results
SCRIPT='/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts_eval/visualize_results_batch.py'
python $SCRIPT --model_name $MODEL_NAME --task_names $EVAL_SETS

echo "Done evaluating $MODEL_NAME on $EVAL_SETS"
