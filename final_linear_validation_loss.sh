#!/bin/bash
# This script executes the PyTorch-based GPU simulation.
export XAI_API_KEY=your_key
XAI_API_KEY=$XAI_API_KEY ./.venv_gpu/bin/python self_discovery.py --final --manual-theories-file linear_signal_loss.py 
