#!/bin/bash

# Function to check if a path is already mounted
is_mounted() {
    mountpoint -q "$1"
    return $?
}

# Mount results directory if not already mounted
if ! is_mounted ~/Projects/results_hyak; then
    echo "Mounting results directory..."
    sshfs hyak:/gscratch/weirdlab/memmelma/simvla/pick_data_gen/results ~/Projects/results_hyak
    if [ $? -eq 0 ]; then
        echo "Results directory mounted successfully"
    else
        echo "Failed to mount results directory"
    fi
else
    echo "Results directory is already mounted"
fi
