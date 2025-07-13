#!/bin/bash
# Run Linear Signal Loss theory with comparisons to defaults

# Go to project root
cd "$(dirname "$0")/../.."

# Run this theory
./run_theory.sh theories/linear_signal_loss "$@" 