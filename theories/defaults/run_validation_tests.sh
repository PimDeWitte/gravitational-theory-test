#!/bin/bash
# Quick validation test runner
# This runs a quick test with reduced steps to verify everything works

cd "$(dirname "$0")/.."

# Run with test mode (reduced steps)
./run_theory.sh --test 
