#!/bin/bash
# Generic theory runner script
# Usage: ./run_theory.sh [theory_dir] [options]
#
# Examples:
#   ./run_theory.sh theories/linear_signal_loss              # Run a specific theory
#   ./run_theory.sh theories/linear_signal_loss --final      # Run with high precision
#   ./run_theory.sh                                          # Run defaults only
#   ./run_theory.sh --skip-defaults theories/my_theory       # Skip default comparisons

set -e

# Default values
THEORY_DIRS=""
THEORY_DIR=""
SKIP_DEFAULTS=false
SKIP_VALIDATIONS=false
SELF_DISCOVER=false
INITIAL_PROMPT=""
API_PROVIDER="grok"
EXTRA_ARGS=""
VALIDATE_BASELINES=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-defaults) SKIP_DEFAULTS=true; shift;;
        --skip-validations) SKIP_VALIDATIONS=true; shift;;
        --self-discover) SELF_DISCOVER=true; shift;;
        --initial-prompt) INITIAL_PROMPT="$2"; shift 2;;
        --api-provider) API_PROVIDER="$2"; shift 2;;
        --final) FINAL=true; shift;;
        --test) TEST=true; shift;;
        --cpu-f64) CPU_F64=true; shift;;
        --validate-observations) VALIDATE=true; shift;;
        --validate-baselines)
            VALIDATE_BASELINES=true
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            if [ -z "$THEORY_DIR" ]; then
                THEORY_DIR="$1"
            else
                EXTRA_ARGS+=" $1"
            fi
            shift
            ;;
    esac
done

# Set up theory directories
if [ -n "$THEORY_DIR" ]; then
    # If a specific theory directory was provided, use it
    if [ "$SKIP_DEFAULTS" = true ]; then
        THEORY_DIRS="$THEORY_DIR"
    else
        THEORY_DIRS="theories/defaults $THEORY_DIR"
    fi
else
    # No specific theory directory, just use defaults
    THEORY_DIRS="theories/defaults"
fi

# Check Python environment
if [[ -f .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
elif [[ -f .venv_gpu/bin/python ]]; then
    # Legacy support - warn about old environment
    echo "WARNING: Found old .venv_gpu environment. Please run ./setup.sh to create unified environment."
    PYTHON=.venv_gpu/bin/python
elif [[ -f .venv_cpu/bin/python ]]; then
    # Legacy support - warn about old environment
    echo "WARNING: Found old .venv_cpu environment. Please run ./setup.sh to create unified environment."
    PYTHON=.venv_cpu/bin/python
else
    PYTHON=python3
fi
echo "Using Python: $PYTHON"

echo "=================================="
echo "Theory Runner"
echo "=================================="
echo "Theory directories: $THEORY_DIRS"
echo "Skip defaults: $SKIP_DEFAULTS"
echo "Skip validations: $SKIP_VALIDATIONS"
echo "Self-discovery: $SELF_DISCOVER"
if [[ "$SELF_DISCOVER" == "true" ]]; then
    echo "API provider: $API_PROVIDER"
    if [[ -n "$INITIAL_PROMPT" ]]; then
        echo "Initial prompt: $INITIAL_PROMPT"
    fi
fi
echo "Extra args: $EXTRA_ARGS"
echo "Python: $PYTHON"
echo "Validate baselines: $VALIDATE_BASELINES"
echo "=================================="

# Ensure cache directory exists
mkdir -p cache

# Build the command
python_cmd="$PYTHON self_discovery.py --theory-dirs $THEORY_DIR"

# Add flags
if [ "$TEST" = true ]; then
    python_cmd="$python_cmd --test"
fi

if [ "$CPU_F64" = true ]; then
    python_cmd="$python_cmd --cpu-f64"
fi

if [ "$VALIDATE" = true ]; then
    python_cmd="$python_cmd --validate-observations"
fi

if [ "$VALIDATE_BASELINES" = true ]; then
    python_cmd="$python_cmd --validate-baselines"
fi

if [ -n "$VERBOSE" ]; then
    python_cmd="$python_cmd $VERBOSE"
fi

# Run the simulation
echo "$python_cmd"
eval "$python_cmd"

# Run validations if not skipped
if [[ "$SKIP_VALIDATIONS" == "false" ]]; then
    echo ""
    echo "=================================="
    echo "Running Observational Validations"
    echo "=================================="
    
    # Use the existing Python runner to run validations
    $PYTHON run_validations.py --theory-dirs $THEORY_DIRS --device mps --dtype float32 $(if [ "$VALIDATE_BASELINES" = true ]; then echo "--validate-baselines"; fi)
fi

echo ""
echo "=================================="
echo "Theory run complete!"
echo "==================================" 