#!/bin/bash

# Activate conda environment
conda activate tf-ssd

# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# Run the 'func' command with the specified port
func start --port 7073