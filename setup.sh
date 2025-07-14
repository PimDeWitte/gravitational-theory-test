#!/bin/bash
# Unified setup script for gravity_compression project
# Creates a single environment that supports both CPU and GPU (MPS) execution

# Function for clear, colored output
print_message() {
    MESSAGE=$1; COLOR=$2
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
    if [ "$COLOR" == "green" ]; then echo -e "${GREEN}${MESSAGE}${NC}";
    elif [ "$COLOR" == "yellow" ]; then echo -e "${YELLOW}${MESSAGE}${NC}";
    elif [ "$COLOR" == "red" ]; then echo -e "${RED}${MESSAGE}${NC}";
    else echo "$MESSAGE"; fi
}

# --- Step 1: Clean up old environments ---
print_message "▶ Setting up unified environment for gravity_compression..." "yellow"
print_message "  Cleaning up old environments..." "yellow"

for env in .venv .venv_gpu .venv_cpu .venv_m3; do
    if [ -d "$env" ]; then
        print_message "  Removing old environment: $env" "yellow"
        rm -rf "$env"
    fi
done

# --- Step 2: Install uv if needed ---
if ! command -v uv &> /dev/null; then
    print_message "  Installing 'uv' package manager..." "yellow"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Step 3: Create unified environment ---
print_message "\n▶ Creating unified Python virtual environment..." "yellow"
uv venv -p python3 .venv
print_message "✔ Virtual environment created." "green"

PYTHON_EXEC="./.venv/bin/python"

# --- Step 4: Install PyTorch with MPS support ---
print_message "\n▶ Installing PyTorch (with Apple Silicon GPU support)..." "yellow"
uv pip install --python $PYTHON_EXEC torch
if [ $? -ne 0 ]; then 
    print_message "✖ Failed to install PyTorch." "red"
    exit 1
fi

# Verify PyTorch installation
TORCH_VERSION=$($PYTHON_EXEC -c "import torch; print(torch.__version__)" 2>&1)
if [ $? -eq 0 ]; then
    print_message "✔ PyTorch ${TORCH_VERSION} installed successfully." "green"
else
    print_message "✖ PyTorch installation verification failed." "red"
    exit 1
fi

# --- Step 5: Install core dependencies ---
print_message "\n▶ Installing core scientific computing packages..." "yellow"
uv pip install --python $PYTHON_EXEC numpy scipy matplotlib requests sympy
if [ $? -ne 0 ]; then 
    print_message "✖ Failed to install core packages." "red"
    exit 1
fi
print_message "✔ Core packages installed." "green"

# --- Step 6: Install pulsar timing packages (optional but included) ---
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
# For accelerated covariance
uv pip install --python $PYTHON_EXEC git+https://github.com/nanograv/enterprise.git@fastshermanmorrison
if [ $? -ne 0 ]; then 
    print_message "⚠ Some pulsar packages may have failed (optional)." "yellow"
else
    print_message "✔ Pulsar timing packages installed." "green"
fi

# --- Step 7: Comprehensive verification ---
print_message "\n▶ Verifying installation..." "yellow"

$PYTHON_EXEC -c "
import torch
import sys
print('='*60)
print(f'Python version: {sys.version.split()[0]}')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS (GPU) available: {torch.backends.mps.is_available()}')
print(f'Default device: {\"mps\" if torch.backends.mps.is_available() else \"cpu\"}')
print('='*60)
print('✓ Environment supports both CPU and GPU execution!')
print('  • Use --cpu-f64 flag for high-precision CPU mode')
print('  • Default mode uses GPU (MPS) if available')
"

# --- Step 8: Update run scripts ---
print_message "\n▶ Updating run scripts..." "yellow"

# Create simplified run.sh
cat > run.sh << 'EOL'
#!/bin/bash
# Simplified run script using unified environment
source .venv/bin/activate
./run_theory.sh "$@"
EOL
chmod +x run.sh

# Update run_gpu.sh to use unified env
cat > run_gpu.sh << 'EOL'
#!/bin/bash
# Legacy compatibility - redirects to unified environment
echo "Note: Using unified environment (.venv)"
./.venv/bin/python self_discovery.py "$@"
EOL
chmod +x run_gpu.sh

print_message "✔ Run scripts updated." "green"

# --- Final summary ---
print_message "\n════════════════════════════════════════════════════════════════" "green"
print_message "✔ Unified setup complete!" "green"
print_message "════════════════════════════════════════════════════════════════" "green"
print_message "\nEnvironment: .venv (unified for both CPU and GPU)" "yellow"
print_message "PyTorch: ${TORCH_VERSION}" "yellow"
print_message "\nUsage:" "yellow"
print_message "  • Run with GPU (default): ./run_theory.sh" "green"
print_message "  • Run with CPU (high precision): ./run_theory.sh --cpu-f64" "green"
print_message "  • Activate manually: source .venv/bin/activate" "green"
print_message "\nAll old environments (.venv_cpu, .venv_gpu, etc.) have been removed." "yellow"
print_message "The new unified environment supports both execution modes." "yellow" 