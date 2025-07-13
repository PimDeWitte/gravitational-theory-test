#!/bin/bash
# This script automates the setup for the PyTorch-based GPU simulation.
# (V12: Adds matplotlib for plotting and removes unused torchdiffeq)

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
print_message "▶ Starting Apple M-series environment setup for PyTorch GPU simulation..." "yellow"
VENV_NAME=".venv_gpu"
PYTHON_EXEC="./${VENV_NAME}/bin/python"

if [ -d "$VENV_NAME" ]; then
    print_message "  Found old environment. Removing './${VENV_NAME}'." "yellow"
    rm -rf "$VENV_NAME"
fi

# --- Step 2: Install uv ---
if ! command -v uv &> /dev/null; then
    print_message "  Installing 'uv' package manager..." "yellow"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to the path for the current session
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Step 3: Create Environment and Install Packages ---
print_message "\n▶ Creating Python virtual environment..." "yellow"
uv venv -p python3 $VENV_NAME
print_message "✔ Virtual environment created." "green"

print_message "\n▶ Installing PyTorch, Matplotlib, and dependencies..." "yellow"
# Installs torch, numpy, scipy, and matplotlib for plotting
uv pip install --python $PYTHON_EXEC torch numpy scipy matplotlib requests
if [ $? -ne 0 ]; then print_message "✖ Failed to install packages." "red"; exit 1; fi
print_message "✔ Packages installed." "green"

# --- Step 3b: Install Pulsar Timing Analysis Dependencies ---
print_message "\n▶ Installing pulsar timing analysis dependencies..." "yellow"
# Core data analysis
uv pip install --python $PYTHON_EXEC h5py astropy pandas
# Pulsar timing packages
uv pip install --python $PYTHON_EXEC pint-pulsar enterprise-pulsar libstempo
# SPICE for barycentric corrections
uv pip install --python $PYTHON_EXEC spiceypy
# MCMC and statistical analysis
uv pip install --python $PYTHON_EXEC emcee corner dynesty nestle
# Additional NANOGrav tools
uv pip install --python $PYTHON_EXEC PTMCMCSampler hasasia la_forge
# For accelerated covariance (fastshermanmorrison branch)
uv pip install --python $PYTHON_EXEC git+https://github.com/nanograv/enterprise.git@fastshermanmorrison
if [ $? -ne 0 ]; then print_message "⚠ Some pulsar packages may have failed. Check output above." "yellow"; fi
print_message "✔ Pulsar timing packages installed." "green"

# --- Step 4: Download SPICE Kernels ---
print_message "\n▶ Downloading SPICE kernels for barycentric corrections..." "yellow"
$PYTHON_EXEC -c "
import spiceypy as spice
import os
# Download required kernels for Earth barycenter (NAIF release N0067)
kernel_dir = os.path.expanduser('~/.spice_kernels')
os.makedirs(kernel_dir, exist_ok=True)
print(f'SPICE kernels will be stored in: {kernel_dir}')
# Note: Manual download of specific kernels may be required
"
print_message "✔ SPICE kernel directory created. Manual download of N0067 may be required." "yellow"

# --- Step 5: Verification ---
print_message "\n▶ Verifying environment integrity..." "yellow"
print_message "  DEBUG: Checking for PyTorch MPS (Metal) device..."
if ! $PYTHON_EXEC -c "import torch; exit(0) if torch.backends.mps.is_available() else exit(1)"; then
    print_message "✖ ERROR: PyTorch installed but MPS device not available." "red"
    exit 1
fi
print_message "✔ PyTorch MPS (Metal) GPU device confirmed." "green"

# Verify pulsar packages
print_message "  DEBUG: Checking pulsar timing packages..."
$PYTHON_EXEC -c "
import sys
packages = ['h5py', 'astropy', 'pint', 'enterprise', 'spiceypy', 'emcee']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('All core pulsar packages verified.')
"
if [ $? -ne 0 ]; then 
    print_message "⚠ Some pulsar packages are missing. Check installation output." "yellow"
else
    print_message "✔ Pulsar timing packages verified." "green"
fi

# --- Step 6: Generate the 'run_gpu.sh' script ---
print_message "\n▶ Generating 'run_gpu.sh' to execute the PyTorch script..." "yellow"
cat > run_gpu.sh << EOL
#!/bin/bash
# This script executes the PyTorch-based GPU simulation.
./${VENV_NAME}/bin/python sim_gpu.py
EOL
chmod +x run_gpu.sh
print_message "✔ 'run_gpu.sh' created." "green"

# --- End of Script ---
print_message "\n▶ Setup complete. Use './run_gpu.sh' to start the GPU simulation." "green"
print_message "  For pulsar timing analysis, activate the environment: source ${VENV_NAME}/bin/activate" "yellow"
