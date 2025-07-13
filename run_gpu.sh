#!/bin/bash
# This script executes the PyTorch-based GPU simulation.
# It ensures PyTorch is available before running.

# Verify PyTorch is installed
if ! ./.venv_gpu/bin/python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Please run ./setup_gpu.sh first."
    exit 1
fi

# Run the simulation
./.venv_gpu/bin/python sim_gpu.py
