#!/bin/bash
# This script executes the PyTorch-based GPU simulation.
export XAI_API_KEY=xai-3oDgWj3Cp1cTAWs3n7wWmLRx6s8CbNh0k7pJaNeZ68mHH37PSf2bbrasjh0Cyk9W4g0sZQYmWXoPVvEm
XAI_API_KEY=$XAI_API_KEY ./.venv_gpu/bin/python self_discovery.py --final --manual-theories-file linear_signal_loss.py 
