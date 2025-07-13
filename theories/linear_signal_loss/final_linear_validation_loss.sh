#!/bin/bash
# This script runs final validation for Linear Signal Loss theory
export XAI_API_KEY=your_key

# Run final validation with Linear Signal Loss theory directory
XAI_API_KEY=$XAI_API_KEY ../../.venv_gpu/bin/python ../../self_discovery.py --final --theory-dirs theories/linear_signal_loss 
