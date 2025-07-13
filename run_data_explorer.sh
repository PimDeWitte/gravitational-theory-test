#!/bin/bash
# Script to run the data explorer with proper Python environment

# Exit the virtual environment if we're in one
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Exiting virtual environment..."
    deactivate 2>&1 || true
fi

# Use the pyenv Python which has h5py installed
PYTHON_CMD="/Users/p/.pyenv/versions/3.12.7/bin/python3"

# Check if h5py is available
echo "Checking for h5py installation..."
$PYTHON_CMD -c "import h5py; print('h5py found successfully')" 
if [ $? -ne 0 ]; then
    echo "Error: h5py not found. Please install it with:"
    echo "  $PYTHON_CMD -m pip install h5py"
    exit 1
fi

# Run the data explorer
echo "Running data explorer..."
$PYTHON_CMD explore_data.py 2>&1 