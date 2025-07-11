#!/bin/bash
# This script executes the PyTorch-based GPU simulation.
export XAI_API_KEY=your_key
XAI_API_KEY=$XAI_API_KEY ./.venv_gpu/bin/python self_discovery.py --test --self-discover --initial-prompt="look at theories found in game engine code and compare to einstein's field equations and death bed scribbles"
