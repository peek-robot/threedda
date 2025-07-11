#!/bin/bash


SPLIT=$1
ROOT_DIR="/lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/data/"
TASK="llarva"
OUT_DIR="$ROOT_DIR/$TASK"

CMD="source /lustre/fsw/portfolios/nvr/users/mmemmel/miniforge3/bin/activate roboverse && \
    python /lustre/fs12/portfolios/nvr/users/mmemmel/projects/vila/convert_roboverse/llarva_to_llava.py --split $SPLIT"

export NCCL_ASYNC_ERROR_HANDLING=1

read -p "Do you want to remove old files? (y/n): " answer
if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    rm $OUT_DIR/${SPLIT}_llarva*.jsonl
    rm $OUT_DIR/${SPLIT}_llarva*.json
    rm $OUT_DIR/${SPLIT}.txt
    echo "Files removed."
else
    echo "NO files removed."
fi

for (( i=0; i<100; i++ ))
do
    echo
    echo "#######################################################"
    echo
    
    # creates file and deletes once done
    srun -A nvr_srl_simpler \
         -J llarva \
         --cpus-per-gpu 16 \
         --gpus 1 \
         --partition polar,polar3,polar4,grizzly,interactive,interactive_singlenode \
         --time 04:00:00 \
         --unbuffered \
         bash -c "$CMD" 2>&1 | tee -a $OUT_DIR/terminal_$SPLIT.log
    sleep 1m

    # repeat until file no longer exists
    if [ ! -e $OUT_DIR/$SPLIT.txt ]; then
        echo "done"
        break
    fi

done

echo "Done."
