MODEL_NAME=${1:-"VILA1.5-3b-rlbench1000_train1000-e10"}
QUESTION_NAME=${2:-"colosseum_test_375_256_sketch_v5_eval"}
CUDA_VISIBLE_DEVICES=${3:-"0,1,2,3,4,5,6,7"}
CHECKPOINTDIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/checkpoints/finetuned/vila"
MODEL_PATH="$CHECKPOINTDIR/$MODEL_NAME"

echo $MODEL_NAME
echo $QUESTION_NAME
echo $CUDA_VISIBLE_DEVICES

DATASET_ROOT="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data"

# check if llarva in question name
if [[ $QUESTION_NAME == *llarva* ]]; then
    QUESTION_NAME="llarva"
fi
if [[ $QUESTION_NAME == "llarva" ]]; then
    IMAGE_FOLDER=${DATASET_ROOT}/${QUESTION_NAME}/images
    QUESTION_FILE=${DATASET_ROOT}/${QUESTION_NAME}/test_llarva_vqa.jsonl
# else is custom dataset
else
    IMAGE_FOLDER=${DATASET_ROOT}/${QUESTION_NAME}/images
    # QUESTION_FILE=${DATASET_ROOT}/${QUESTION_NAME}${QUESTION_NAME}_vqa.jsonl
    QUESTION_FILE=${DATASET_ROOT}/${QUESTION_NAME}/test_${QUESTION_NAME}_vqa.jsonl
    # IMAGE_FOLDER=${DATASET_ROOT}/${QUESTION_NAME}/test/images
    # QUESTION_FILE=${DATASET_ROOT}/${QUESTION_NAME}/test/${QUESTION_NAME}_vqa.jsonl
fi

ANSWER_DIR="$MODEL_PATH/eval"
echo $QUESTION_FILE
echo $IMAGE_FOLDER
echo $MODEL_PATH
echo $ANSWER_FILE
CONV_MODE=vicuna_v1

mkdir -p $MODEL_PATH/eval

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "Launching process $IDX on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file ${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode $CONV_MODE \
        --max_new_tokens 1024 &
        # --generation-config '{"max_new_tokens": 1024}' &
done

wait

output_file=$ANSWER_DIR/$QUESTION_NAME.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_file="${ANSWER_DIR}/${CHUNKS}_${IDX}.jsonl"
    if [[ -f "$chunk_file" ]]; then
        cat "$chunk_file" >> "$output_file"
        echo "" >> "$output_file"  # Add a newline after each file
    else
        echo "Warning: Chunk file $chunk_file not found!"
    fi
done

