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
THEORY_DIRS="theories/defaults"
SKIP_DEFAULTS=false
SKIP_VALIDATIONS=false
SELF_DISCOVER=false
INITIAL_PROMPT=""
API_PROVIDER="grok"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-defaults)
            SKIP_DEFAULTS=true
            shift
            ;;
        --skip-validations)
            SKIP_VALIDATIONS=true
            shift
            ;;
        --self-discover)
            SELF_DISCOVER=true
            shift
            ;;
        --initial-prompt)
            INITIAL_PROMPT="$2"
            shift 2
            ;;
        --api-provider)
            API_PROVIDER="$2"
            shift 2
            ;;
        --final|--test|--cpu-f64)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        theories/*)
            # Theory directory path
            if [[ "$SKIP_DEFAULTS" == "false" ]]; then
                THEORY_DIRS="theories/defaults $1"
            else
                THEORY_DIRS="$1"
            fi
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [theory_dir] [options]"
            echo "Options:"
            echo "  --skip-defaults      Don't compare against default theories"
            echo "  --skip-validations   Don't run observational validations"
            echo "  --self-discover      Enable AI-powered theory discovery mode"
            echo "  --initial-prompt     Initial prompt for theory generation"
            echo "  --api-provider       API provider (grok, gemini, openai, anthropic)"
            echo "  --final             Run with high precision"
            echo "  --test              Run with reduced steps"
            echo "  --cpu-f64           Use CPU with float64"
            exit 1
            ;;
    esac
done

# Check Python environment
if [[ -f .venv_gpu/bin/python ]]; then
    PYTHON=.venv_gpu/bin/python
elif [[ -f .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python3
fi

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
echo "=================================="

# Ensure cache directory exists
mkdir -p cache

# Build command
CMD="$PYTHON self_discovery.py --theory-dirs $THEORY_DIRS $EXTRA_ARGS"

# Add self-discovery options if enabled
if [[ "$SELF_DISCOVER" == "true" ]]; then
    CMD="$CMD --self-discover"
    
    # Check for API key
    if [[ "$API_PROVIDER" == "grok" && -z "$XAI_API_KEY" ]]; then
        echo "Error: XAI_API_KEY environment variable required for self-discovery with Grok"
        exit 1
    elif [[ "$API_PROVIDER" == "gemini" && -z "$GEMINI_API_KEY" ]]; then
        echo "Error: GEMINI_API_KEY environment variable required for self-discovery with Gemini"
        exit 1
    elif [[ "$API_PROVIDER" == "openai" && -z "$OPENAI_API_KEY" ]]; then
        echo "Error: OPENAI_API_KEY environment variable required for self-discovery with OpenAI"
        exit 1
    elif [[ "$API_PROVIDER" == "anthropic" && -z "$ANTHROPIC_API_KEY" ]]; then
        echo "Error: ANTHROPIC_API_KEY environment variable required for self-discovery with Anthropic"
        exit 1
    fi
    
    CMD="$CMD --api-provider $API_PROVIDER"
    
    if [[ -n "$INITIAL_PROMPT" ]]; then
        CMD="$CMD --initial-prompt \"$INITIAL_PROMPT\""
    fi
fi

# Run the command
eval $CMD

# Run validations if not skipped
if [[ "$SKIP_VALIDATIONS" == "false" ]]; then
    echo ""
    echo "=================================="
    echo "Running Observational Validations"
    echo "=================================="
    
    # Use the existing Python runner to run validations
    $PYTHON -u -c "
import sys
import os
import importlib.util
import traceback

# Add current directory to Python path for imports
sys.path.insert(0, os.getcwd())

# Get theory directories from command line
theory_dirs = '$THEORY_DIRS'.split()
print(f'Theory directories to validate: {theory_dirs}')

# Import base_theory
try:
    import base_theory
    print('Successfully loaded base_theory.py')
except Exception as e:
    print(f'ERROR loading base_theory.py: {e}')
    traceback.print_exc()
    sys.exit(1)

# Load validation classes
validation_modules = {}
for validation_file in ['base_validation', 'pulsar_timing', 'mercury_perihelion', 'cassini_ppn']:
    try:
        spec = importlib.util.spec_from_file_location(
            validation_file, 
            f'theories/defaults/validations/{validation_file}.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        validation_modules[validation_file] = module
        print(f'Loaded validation module: {validation_file}')
    except Exception as e:
        print(f'ERROR loading {validation_file}: {e}')
        traceback.print_exc()

# Load theories from directories (excluding baseline theories)
def load_theories_from_dir(theory_dir):
    theories = []
    source_dir = os.path.join(theory_dir, 'source')
    print(f'Looking for theories in: {source_dir}')
    
    if os.path.exists(source_dir):
        # Get all .py files in the source directory
        theory_files = [f for f in os.listdir(source_dir) if f.endswith('.py') and not f.startswith('__')]
        
        if not theory_files:
            print(f'  No theory files found in source directory')
        
        for filename in theory_files:
            filepath = os.path.join(source_dir, filename)
            print(f'  Checking file: {filename}')
            
            try:
                # Use unique module names to avoid conflicts
                module_name = f'theory_module_{theory_dir.replace(\"/\", \"_\")}_{filename[:-3]}'
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get all classes that inherit from GravitationalTheory
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, base_theory.GravitationalTheory) and attr != base_theory.GravitationalTheory:
                        try:
                            instance = attr()
                            theories.append(instance)
                            print(f'    Found theory: {attr.__name__}')
                        except Exception as e:
                            print(f'    ERROR instantiating {attr.__name__}: {e}')
            except Exception as e:
                print(f'  ERROR loading {filename}: {e}')
                traceback.print_exc()
    else:
        print(f'  No source directory found in {theory_dir}')
    
    # Note about baseline theories
    baseline_dir = os.path.join(theory_dir, 'baselines')
    if os.path.exists(baseline_dir):
        baseline_files = [f for f in os.listdir(baseline_dir) if f.endswith('.py') and not f.startswith('__')]
        if baseline_files:
            print(f'  Note: Found {len(baseline_files)} baseline theories (not validated):')
            for f in baseline_files:
                print(f'    - {f}')
    
    return theories

# Load all theories
all_theories = {}
for theory_dir in theory_dirs:
    print(f'\nLoading theories from: {theory_dir}')
    theories = load_theories_from_dir(theory_dir)
    if theories:
        all_theories[theory_dir] = theories
        print(f'  Total theories loaded: {len(theories)}')
    else:
        print(f'  No theories found in {theory_dir}')

print(f'\nTotal theory directories with theories: {len(all_theories)}')

# Initialize validators
try:
    validators = [
        validation_modules['pulsar_timing'].PulsarTimingValidation(),
        validation_modules['mercury_perihelion'].MercuryPerihelionValidation(),
        validation_modules['cassini_ppn'].CassiniPPNValidation(),
    ]
    print(f'Initialized {len(validators)} validators')
except Exception as e:
    print(f'ERROR initializing validators: {e}')
    traceback.print_exc()
    validators = []

# Run validations
if not all_theories:
    print('\nNo theories found to validate!')
else:
    for theory_dir, theories in all_theories.items():
        print(f'\nValidating theories from: {theory_dir}')
        print('-' * 50)
        
        for theory in theories:
            print(f'\nTheory: {theory.name}')
            
            for validator in validators:
                try:
                    result = validator.validate(theory)
                    status = 'PASS' if result['pass'] else 'FAIL'
                    print(f'  {validator.name}: {status}')
                    print(f'    Observed: {result[\"observed\"]:.6f}')
                    print(f'    Predicted: {result[\"predicted\"]:.6f}')
                    print(f'    Error: {result[\"error\"]:.6f}')
                    
                    if 'details' in result and 'error' in result['details']:
                        print(f'    Error: {result[\"details\"][\"error\"]}')
                except Exception as e:
                    print(f'  {validator.name}: ERROR - {str(e)}')
                    traceback.print_exc()
"
fi

echo ""
echo "=================================="
echo "Theory run complete!"
echo "==================================" 