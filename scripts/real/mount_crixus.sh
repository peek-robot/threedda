#!/bin/bash

# Function to check if a path is already mounted
is_mounted() {
    mountpoint -q "$1"
    return $?
}

# Mount results directory if not already mounted
if ! is_mounted ~/Projects/problem_reduction/results; then
    echo "Mounting results directory..."
    sshfs crixus:/home/memmelma/Projects/robotic/results ~/Projects/problem_reduction/results
    if [ $? -eq 0 ]; then
        echo "Results directory mounted successfully"
    else
        echo "Failed to mount results directory"
    fi
else
    echo "Results directory is already mounted"
fi

# Mount data directory if not already mounted
if ! is_mounted ~/Projects/problem_reduction/data; then
    echo "Mounting data directory..."
    sshfs crixus:/home/memmelma/Projects/robotic/gifs_curobo ~/Projects/problem_reduction/data
    if [ $? -eq 0 ]; then
        echo "Data directory mounted successfully"
    else
        echo "Failed to mount data directory"
    fi
else
    echo "Data directory is already mounted"
fi
