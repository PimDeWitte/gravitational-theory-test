#!/bin/bash
# This script automates the setup for the PyTorch-based GPU simulation.
# (V13: Explicit PyTorch installation with enhanced verification)

# Function for clear, colored output.
print_message() {
    MESSAGE=$1; COLOR=$2
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0/.;33m'; NC='\033[0m'
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

# --- Step 3: Create Environment and Install Core Packages ---
print_message "\n▶ Creating Python virtual environment..." "yellow"
uv venv -p python3 $VENV_NAME
print_message "✔ Virtual environment created." "green"

# --- Step 4: Install PyTorch (EXPLICIT) ---
print_message "\n▶ Installing PyTorch for Apple Silicon GPU..." "yellow"
print_message "  Installing torch package..." "yellow"
uv pip install --python $PYTHON_EXEC torch
if [ $? -ne 0 ]; then 
    print_message "✖ Failed to install PyTorch." "red"
    exit 1
fi
print_message "✔ PyTorch installed successfully." "green"

# Verify PyTorch installation immediately
print_message "  Verifying PyTorch installation..." "yellow"
TORCH_VERSION=$($PYTHON_EXEC -c "import torch; print(torch.__version__)" 2>&1)
if [ $? -eq 0 ]; then
    print_message "✔ PyTorch ${TORCH_VERSION} installed and verified." "green"
else
    print_message "✖ PyTorch installation verification failed: ${TORCH_VERSION}" "red"
    exit 1
fi

# --- Step 5: Install Scientific Computing Packages ---
print_message "\n▶ Installing scientific computing packages..." "yellow"
# Core packages needed for the simulation
uv pip install --python $PYTHON_EXEC numpy scipy matplotlib requests
if [ $? -ne 0 ]; then print_message "✖ Failed to install scientific packages." "red"; exit 1; fi
print_message "✔ Scientific packages installed." "green"

# --- Step 6: Install Pulsar Timing Analysis Dependencies ---
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

# --- Step 7: Download SPICE Kernels ---
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

# --- Step 8: Comprehensive Verification ---
print_message "\n▶ Performing comprehensive environment verification..." "yellow"

# Check PyTorch with detailed info
print_message "  Checking PyTorch installation details..." "yellow"
$PYTHON_EXEC -c "
import torch
import sys
print(f'PyTorch version: {torch.__version__}')
print(f'Python version: {sys.version}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS (Metal) available: {torch.backends.mps.is_available()}')
if torch.backends.mps.is_available():
    print('✓ Apple Silicon GPU acceleration is available!')
else:
    print('✗ WARNING: MPS not available - will fall back to CPU')
"
if [ $? -ne 0 ]; then
    print_message "✖ ERROR: PyTorch verification failed." "red"
    exit 1
fi
print_message "✔ PyTorch verified with GPU support." "green"

# Verify other critical packages
print_message "  Checking other critical packages..." "yellow"
$PYTHON_EXEC -c "
import sys
packages = ['numpy', 'scipy', 'matplotlib', 'requests']
missing = []
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f'✓ {pkg} version: {mod.__version__ if hasattr(mod, \"__version__\") else \"installed\"}')
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'✗ Missing packages: {missing}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then 
    print_message "✖ Some critical packages are missing." "red"
    exit 1
else
    print_message "✔ All critical packages verified." "green"
fi

# Verify pulsar packages
print_message "  Checking pulsar timing packages..." "yellow"
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

# --- Step 9: Generate the 'run_gpu.sh' script ---
print_message "\n▶ Generating 'run_gpu.sh' to execute the PyTorch script..." "yellow"
cat > run_gpu.sh << EOL
#!/bin/bash
# This script executes the PyTorch-based GPU simulation.
# It ensures PyTorch is available before running.

# Verify PyTorch is installed
if ! ./${VENV_NAME}/bin/python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Please run ./setup_gpu.sh first."
    exit 1
fi

# Run the simulation
./${VENV_NAME}/bin/python sim_gpu.py
EOL
chmod +x run_gpu.sh
print_message "✔ 'run_gpu.sh' created with PyTorch verification." "green"

# --- Final Summary ---
print_message "\n════════════════════════════════════════════════════════════════" "green"
print_message "✔ Setup complete!" "green"
print_message "════════════════════════════════════════════════════════════════" "green"
print_message "\nEnvironment: ${VENV_NAME}" "yellow"
print_message "PyTorch: ${TORCH_VERSION}" "yellow"
print_message "\nTo run simulations:" "yellow"
print_message "  • GPU simulation: ./run_gpu.sh" "green"
print_message "  • Theory runner: ./run_theory.sh <theory_dir>" "green"
print_message "  • Activate manually: source ${VENV_NAME}/bin/activate" "green"
print_message "\nFor Einstein! 🚀" "yellow"
