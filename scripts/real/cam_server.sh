#!/bin/bash

# Exit on any error
set -e

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the conda environment and run the script
conda activate foundation_stereo && cd /home/marius/Projects/robotic_sim2real && python fs2r/network/cam_server.py