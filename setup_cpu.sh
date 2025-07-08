#!/bin/bash
# This script automates the setup for the final, robust CPU-based simulation.

# Function for clear, colored output.
print_message() {
    MESSAGE=$1; COLOR=$2
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
    if [ "$COLOR" == "green" ]; then echo -e "${GREEN}${MESSAGE}${NC}";
    elif [ "$COLOR" == "yellow" ]; then echo -e "${YELLOW}${MESSAGE}${NC}";
    elif [ "$COLOR" == "red" ]; then echo -e "${RED}${MESSAGE}${NC}";
    else echo "$MESSAGE"; fi
}

# --- Step 1: Preliminary Checks ---
print_message "▶ Starting environment setup for High-Precision CPU simulation..." "yellow"
VENV_NAME=".venv_cpu"
PYTHON_EXEC="./${VENV_NAME}/bin/python"

if [ -d "$VENV_NAME" ]; then
    print_message "  Found old environment. Removing './${VENV_NAME}'." "yellow"
    rm -rf "$VENV_NAME"
fi

# --- Step 2: Install uv ---
if ! command -v uv &> /dev/null; then
    print_message "  Installing 'uv' package manager..." "yellow"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Step 3: Create Environment and Install Packages ---
print_message "\n▶ Creating Python virtual environment..." "yellow"
uv venv -p python3 $VENV_NAME
print_message "✔ Virtual environment created." "green"

print_message "\n▶ Installing required packages: numpy & scipy..." "yellow"
uv pip install --python $PYTHON_EXEC numpy scipy
if [ $? -ne 0 ]; then print_message "✖ Failed to install packages." "red"; exit 1; fi
print_message "✔ Packages installed." "green"

# --- Step 4: Verification ---
print_message "\n▶ Verifying environment..." "yellow"
if ! $PYTHON_EXEC -c "import numpy; import scipy" &> /dev/null; then
    print_message "✖ ERROR: numpy or scipy failed to import." "red"; exit 1;
fi
print_message "✔ Environment verified successfully." "green"

# --- Step 5: Generate the 'run.sh' script ---
print_message "\n▶ Generating 'run_cpu.sh' to execute the definitive CPU script..." "yellow"
cat > run_cpu.sh << EOL
#!/bin/bash
# This script executes the definitive, high-precision CPU-based simulation.
export VECLIB_MAXIMUM_THREADS=1
./${VENV_NAME}/bin/python run_theories_cpu_definitive.py
EOL
chmod +x run_cpu.sh
print_message "✔ 'run_cpu.sh' created." "green"

# --- End of Script ---
print_message "\n▶ Setup complete. Use './run_cpu.sh' to start the high-precision simulation." "green"