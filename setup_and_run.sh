#!/bin/bash

# This script automates the setup and execution of the gravity compression simulation.
# It is designed to be cross-platform and will handle dependency installation.

# --- Step 1: Define a function for clear, colored output ---
# This makes the script's progress easier to follow for the user.
print_message() {
    # The first argument ($1) is the message text.
    MESSAGE=$1
    # The second argument ($2) is the color.
    COLOR=$2
    # Define color codes for the terminal output.
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    NC='\033[0m' # No Color

    # Check which color was requested and print the message accordingly.
    if [ "$COLOR" == "green" ]; then
        echo -e "${GREEN}${MESSAGE}${NC}"
    elif [ "$COLOR" == "yellow" ]; then
        echo -e "${YELLOW}${MESSAGE}${NC}"
    else
        echo "$MESSAGE"
    fi
}

# --- Step 2: Detect the Operating System ---
# This allows the script to use the correct commands for different platforms.
print_message "▶ Detecting Operating System..." "yellow"
# The 'uname' command provides system information.
OS="$(uname -s)"
# Use a case statement to handle different OS strings.
case "${OS}" in
    # For Linux systems.
    Linux*)     MACHINE=linux;;
    # For macOS systems.
    Darwin*)    MACHINE=mac;;
    # For Windows systems running Git Bash or Cygwin.
    CYGWIN*|MINGW*|MSYS*) MACHINE=windows;;
    # Handle any unknown operating systems.
    *)          MACHINE=unknown;;
esac

# If the OS is unknown, print an error message and exit the script.
if [ "$MACHINE" == "unknown" ]; then
    print_message "Unsupported OS: $OS. Please run on macOS, Linux, or Windows (with Git Bash/WSL)." "red"
    exit 1
fi
print_message "✔ Detected $MACHINE" "green"

# --- Step 3: Check for and Install 'uv' ---
# 'uv' is a fast, modern Python package manager we will use for setup.
print_message "\n▶ Checking for 'uv' package manager..." "yellow"
# The 'command -v' command checks if 'uv' is an executable command in the system's PATH.
if ! command -v uv &> /dev/null; then
    print_message "  'uv' not found. Installing now..." "yellow"
    # Use the appropriate installation command based on the detected OS.
    if [ "$MACHINE" == "windows" ]; then
        # On Windows, we use PowerShell's 'irm' (Invoke-RestMethod) to download and run the installer.
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    else
        # On macOS and Linux, we use 'curl' to download and run the installer shell script.
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    
    # After installation, the shell needs to be sourced to recognize the new command.
    # This section adds the uv executable to the PATH for the current session.
    if [ "$MACHINE" == "windows" ]; then
        # On Windows, the path is typically in the user's cargo bin.
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        # On macOS/Linux, it's in the user's .local/bin.
        export PATH="$HOME/.local/bin:$PATH"
    fi
    print_message "✔ 'uv' installed successfully." "green"
else
    print_message "✔ 'uv' is already installed." "green"
fi

# --- Step 4: Create a Virtual Environment ---
# This creates an isolated space for our Python packages.
VENV_NAME=".venv"
print_message "\n▶ Setting up Python virtual environment in './${VENV_NAME}'..." "yellow"
# Use 'uv venv' to create the virtual environment. The '-p python3' flag ensures it uses Python 3.
uv venv -p python3 $VENV_NAME
print_message "✔ Virtual environment created." "green"

# --- Step 5: Install Dependencies ---
# We need 'numpy' and 'scipy' for the simulation.
print_message "\n▶ Installing dependencies (numpy, scipy)..." "yellow"
# Use 'uv pip install' to install the packages into the virtual environment we just created.
# This is much faster than using standard pip.
uv pip install -p python3 --python ./${VENV_NAME}/bin/python numpy scipy
print_message "✔ Dependencies installed." "green"

# --- Step 6: Run the Simulation ---
# This is the final step where we execute the main Python script.
print_message "\n▶ Running the gravity compression simulation..." "yellow"
# We execute the 'run_all_theories.py' script using the Python interpreter
# that is inside our newly created virtual environment.
./${VENV_NAME}/bin/python run_all_theories.py

# --- End of Script ---
print_message "\n▶ Simulation complete. Script finished." "green"
