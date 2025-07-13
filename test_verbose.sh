#!/bin/bash
# Test script to demonstrate verbose vs non-verbose logging

echo "=========================================="
echo "Testing non-verbose mode (default)"
echo "=========================================="
./run_theory.sh theories/linear_signal_loss --test

echo ""
echo ""
echo "=========================================="
echo "Testing verbose mode"
echo "=========================================="
./run_theory.sh theories/linear_signal_loss --test --verbose 