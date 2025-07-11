#!/bin/bash

BASE_MODEL="Efficient-Large-Model/VILA1.5-3b"
OUT_DIR_BASE="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/checkpoints/finetuned/vila"
SESSION_NAME="slurm_jobs_3"

# Set the base directory where srun_sft.sh and srun_eval.sh exist
BASE_DIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/scripts"

# Flag to enable dry run mode
DRY_RUN=false

# Check for dry-run argument
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run enabled. Commands will be printed but not executed."
fi

# Define dataset combinations from the table (corrected for dataset names)
runs=(
#   "libero_90 libero_10 libero90_finetune"
#   "libero_10 libero_90 libero10_finetune"
#   "libero_90 libero_90 libero90_upper_bound"
#   "libero_10 libero_10 libero10_upper_bound"
#   "libero_90+libero_10 libero_90+libero_10 libero90_libero10_more_data"
  "stack_platforms2_mask stack_platforms2_mask stack_mask"
  "stack_platforms2_mask_path stack_platforms2_mask+stack_platforms2_path stack_mask_path"
  "stack_platforms2_path stack_platforms2_path stack_path"
  "robopoint_1432k_beta+stack_platforms2_mask stack_platforms2_mask robopoint_mask"
  "robopoint_1432k_beta+stack_platforms2_mask_path stack_platforms2_mask+stack_platforms2_path robopoint_mask_path"
  "robopoint_1432k_beta+stack_platforms2_path stack_platforms2_path robopoint_path"
#   "libero_90_lp_rw_ng+libero_90_lp_rw_ng libero_10_lp_rw_ng+libero_10_lp_rw_ng rewording_task_generalization_1"
#   "libero_90_lp_rw_ng libero_10_lp_rw_ng+libero_10_lp_rw_ng+libero_90_lp_rw_ng rewording_baseline_1"
#   "libero_10_lp_rw_ng+libero_10_lp_rw_ng libero_90_lp_rw_ng+libero_90_lp_rw_ng rewording_task_generalization_2"
#   "libero_90_lp_rw_ng libero_90_lp_rw_ng+libero_90_lp_rw_ng+libero_10_lp_rw_ng rewording_baseline_2"
#   "libero_90_lp_rw_ng+libero_90_lp_rw_ng+libero_10_lp_rw_ng+libero_10_lp_rw_ng libero_90_lp_rw_ng+libero_90_lp_rw_ng+libero_10_lp_rw_ng+libero_10_lp_rw_ng more_data_better_performance"
)

# Create a new tmux session in detached mode
if [[ "$DRY_RUN" == false ]]; then
    tmux new-session -d -s $SESSION_NAME
fi

# Loop over dataset configurations and create a new tmux window for each training+evaluation run
for run in "${runs[@]}"; do
    set -- $run
    TRAINDATASETS=$1
    EVALDATASETS=$2
    OUT_NAME=$3
    OUT_DIR="$OUT_DIR_BASE/$OUT_NAME"

    # CMD="sh $BASE_DIR/srun_sft.sh $BASE_MODEL $TRAINDATASETS 1 1e-5 $OUT_DIR"
    # CMD="sh $BASE_DIR/srun_sft.sh $BASE_MODEL $TRAINDATASETS 1 1e-5 $OUT_DIR && sh $BASE_DIR/srun_eval.sh $OUT_NAME $EVALDATASETS"
    CMD="sh $BASE_DIR/srun_eval.sh $OUT_NAME $EVALDATASETS"

    if [[ "$DRY_RUN" == true ]]; then
        echo $CMD
        echo "####"
    else
        echo "Starting training and evaluation for $OUT_NAME..."
        tmux new-window -t $SESSION_NAME -n "$OUT_NAME" "$CMD; exec bash"
        sleep 2  # Avoid launching all jobs at once
    fi
done

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. No jobs were actually started."
else
    echo "All jobs submitted inside a tmux session. Use 'tmux attach-session -t $SESSION_NAME' to monitor."
fi