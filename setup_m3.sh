#!/bin/bash
# This script automates the setup for the Apple Silicon (M3) environment.
# (Corrected version for uv sync syntax)

# Function for clear, colored output.
print_message() {
    MESSAGE=$1; COLOR=$2
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
    if [ "$COLOR" == "green" ]; then echo -e "${GREEN}${MESSAGE}${NC}";
    elif [ "$COLOR" == "yellow" ]; then echo -e "${YELLOW}${MESSAGE}${NC}";
    elif [ "$COLOR" == "red" ]; then echo -e "${RED}${MESSAGE}${NC}";
    else echo "$MESSAGE"; fi
}

# --- Step 1: Preliminary Checks and Cleanup ---
print_message "▶ Starting Apple M3 Max environment setup..." "yellow"
if ! command -v python3 >/dev/null || ! command -v curl >/dev/null; then
    print_message "✖ ERROR: python3 and curl are required. Please install them." "red"
    exit 1
fi

VENV_NAME=".venv_m3"
if [ -d "$VENV_NAME" ]; then
    print_message "  Found old environment. Removing './${VENV_NAME}'." "yellow"
    rm -rf "$VENV_NAME"
fi

# --- Step 2: Install uv and Generate requirements.txt ---
if ! command -v uv &> /dev/null; then
    print_message "  Installing 'uv' package manager..." "yellow"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

print_message "▶ Generating requirements_m3.txt file..." "yellow"
cat > requirements_m3.txt << EOL
numpy
scipy
EOL
print_message "✔ requirements_m3.txt created." "green"

# --- Step 3: Create, Sync, and Verify the Environment ---
print_message "\n▶ Setting up Python virtual environment with 'uv sync'..." "yellow"
# Create a new, empty virtual environment.
uv venv -p python3 $VENV_NAME

# --- PATCHED SYNC COMMAND ---
# This now uses the correct syntax for 'uv sync' by pointing to the
# Python executable within the virtual environment.
uv pip sync requirements_m3.txt --python ./${VENV_NAME}/bin/python

# Verify that the key dependency 'numpy' was actually installed.
print_message "  Verifying installation..." "yellow"
if ! uv pip list --python ./${VENV_NAME}/bin/python | grep -q "numpy"; then
    print_message "✖ ERROR: 'numpy' installation failed." "red"
    print_message "  Please check for errors in the step above."
    exit 1
fi
print_message "✔ Virtual environment created and verified successfully." "green"

# --- Step 4: Generate the 'run.sh' script ---
print_message "\n▶ Generating 'run.sh' for easy execution..." "yellow"
cat > run.sh << EOL
#!/bin/bash
# This script executes the simulation using the correct Apple Silicon environment.
./${VENV_NAME}/bin/python run_all_theories_m3.py
EOL
chmod +x run.sh
print_message "✔ 'run.sh' created." "green"

# --- End of Script ---
print_message "\n▶ Setup complete. Use './run.sh' to start the simulation." "green"
