#!/bin/bash
# This script creates a clean, verified environment and a simple 'run.sh' script.

# Function for clear, colored output.
print_message() {
    MESSAGE=$1; COLOR=$2
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
    if [ "$COLOR" == "green" ]; then echo -e "${GREEN}${MESSAGE}${NC}";
    elif [ "$COLOR" == "yellow" ]; then echo -e "${YELLOW}${MESSAGE}${NC}";
    elif [ "$COLOR" == "red" ]; then echo -e "${RED}${MESSAGE}${NC}";
    else echo "$MESSAGE"; fi
}

# --- Step 1: Preliminary Checks (Dependencies and Cleanup) ---
print_message "▶ Starting environment setup..." "yellow"
# Verify core system tools are present.
if ! command -v python3 >/dev/null || ! command -v curl >/dev/null; then
    print_message "✖ ERROR: python3 and curl are required. Please install them to continue." "red"
    exit 1
fi

VENV_NAME=".venv_gpu"
# PATCH: If an old environment exists, remove it to ensure a clean start.
if [ -d "$VENV_NAME" ]; then
    print_message "  Found old environment. Removing './${VENV_NAME}' for a clean setup." "yellow"
    rm -rf "$VENV_NAME"
fi

# --- Step 2: Install uv and Generate requirements.txt ---
# Install uv if it's not present.
if ! command -v uv &> /dev/null; then
    print_message "  Installing 'uv' package manager..." "yellow"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
# Generate the requirements.txt file on the fly.
cat > requirements.txt << EOL
scipy
# CRITICAL: Adjust this line to match your GPU cluster's CUDA version.
cupy-cuda12x
EOL

# --- Step 3: Create, Sync, and Verify the Environment ---
print_message "\n▶ Creating and syncing virtual environment..." "yellow"
# Create a new, empty virtual environment.
uv venv -p python3 $VENV_NAME
# Use 'uv sync' to install the exact packages from requirements.txt.
uv sync --python ./${VENV_NAME}/bin/python requirements.txt

# PATCH: Verify that the key dependency 'cupy' was actually installed.
print_message "  Verifying installation..." "yellow"
if ! ./${VENV_NAME}/bin/pip list | grep -q "cupy"; then
    print_message "✖ ERROR: 'cupy' installation failed." "red"
    print_message "  Please check your CUDA version and ensure the package in requirements.txt is correct."
    exit 1
fi
print_message "✔ Virtual environment created and verified successfully." "green"

# --- Step 4: PATCH: Generate the 'run.sh' script ---
print_message "\n▶ Generating 'run.sh' for easy execution..." "yellow"
# This heredoc creates a new, simple script to run the simulation.
cat > run.sh << EOL
#!/bin/bash
# This script executes the simulation using the correct Python interpreter.
# Run this file anytime you want to start the analysis.

# Activate the virtual environment and run the python script in one line
# to ensure the correct dependencies are used.
./${VENV_NAME}/bin/python run_all_theories_gpu.py
EOL

# Make the new run script executable.
chmod +x run.sh
print_message "✔ 'run.sh' created." "green"

# --- End of Script ---
print_message "\n▶ Setup complete. You can now use './run.sh' to start the simulation." "green"
