import torch
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind  # For statistical significance

# Import validators and theory
from source.theory import LinearSignalLoss
from ..defaults.validations.mercury_perihelion_validation import MercuryPerihelionValidation
from ..defaults.validations.pulsar_anomaly_validation import PulsarAnomalyValidation
from ..defaults.validations.pulsar_timing_validation import PulsarTimingValidation
from ..defaults.validations.cassini_ppn_validation import CassiniValidation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))  # Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'defaults', 'baselines'))
from schwarzschild import Schwarzschild
from reissner_nordstrom import ReissnerNordstrom

# Constants for reproducibility
GAMMA = 0.75  # Sweet spot from feedback
NUM_RUNS = 10  # For statistics
SEED = 42  # Fixed seed
Q_VALUES = np.logspace(18, 20, 5)  # For RN sensitivity
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

def run_validation(validator, theory, baselines, verbose=False, test=True):
    results = []
    for _ in range(NUM_RUNS):
        result = validator.validate(theory, verbose=verbose, test=test)
        # Compute unification metrics
        loss_gr = abs(result.get('predicted', 0) - baselines['GR'])
        loss_rn = abs(result.get('predicted', 0) - baselines['RN'])
        balance = abs(loss_gr - loss_rn)
        effective_q = np.sqrt(balance / (loss_gr + 1e-10))  # Proxy for emulated charge
        result['loss_gr'] = loss_gr
        result['loss_rn'] = loss_rn
        result['balance'] = balance
        result['effective_q'] = effective_q
        results.append(result)
    return results

def compute_statistics(results):
    """Compute mean, std, p-value for symmetry (loss_gr vs loss_rn)."""
    losses_gr = [r['loss_gr'] for r in results]
    losses_rn = [r['loss_rn'] for r in results]
    mean_balance = np.mean([r['balance'] for r in results])
    std_balance = np.std([r['balance'] for r in results])
    t_stat, p_value = ttest_ind(losses_gr, losses_rn)
    return {
        'mean_balance': mean_balance,
        'std_balance': std_balance,
        'p_symmetry': p_value,  # p>0.05 suggests symmetric (null hypothesis: no difference)
        'interpretation': 'Symmetric (unified) if p>0.05 and mean_balance ≈0'
    }

def plot_symmetry(results, test_name, timestamp):
    balances = [r['balance'] for r in results]
    plt.figure(figsize=(8, 6))
    plt.hist(balances, bins=10, alpha=0.7)
    plt.title(f'Loss Balance Distribution for {test_name} at γ=0.75')
    plt.xlabel('Loss Balance (GR vs RN)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='r', linestyle='--', label='Perfect Symmetry')
    plt.legend()
    plot_path = os.path.join(PLOTS_DIR, f'{test_name}_symmetry_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        'timestamp': timestamp,
        'gamma': GAMMA,
        'num_runs': NUM_RUNS,
        'device': str(DEVICE),
        'dtype': str(DTYPE),
        'tests': {}
    }
    
    theory = LinearSignalLoss(gamma=GAMMA)
    baselines = {
        'GR': Schwarzschild(),
        'RN': ReissnerNordstrom(Q=1e19)  # Baseline Q from feedback
    }
    
    validators = {
        'mercury': MercuryPerihelionValidation(device=DEVICE, dtype=DTYPE),
        'pulsar_anomaly': PulsarAnomalyValidation(device=DEVICE, dtype=DTYPE),
        'pulsar_timing': PulsarTimingValidation(device=DEVICE, dtype=DTYPE),
        'cassini': CassiniValidation(device=DEVICE, dtype=DTYPE)
    }
    
    for test_name, validator in validators.items():
        print(f"Running {test_name} validation...")
        results = run_validation(validator, theory, baselines)
        stats = compute_statistics(results)
        plot = plot_symmetry(results, test_name, timestamp)
        summary['tests'][test_name] = {
            'results': results,
            'stats': stats,
            'plot': plot
        }
    
    # Sensitivity to Q
    print("Running Q sensitivity analysis...")
    q_sensitivity = []
    for Q in Q_VALUES:
        baselines['RN'] = ReissnerNordstrom(Q=Q)
        results = run_validation(validators['pulsar_anomaly'], theory, baselines)  # Use pulsar as key test
        stats = compute_statistics(results)
        q_sensitivity.append({'Q': Q, 'balance': stats['mean_balance']})
    summary['q_sensitivity'] = q_sensitivity
    
    # Save results
    json_path = os.path.join(RESULTS_DIR, f'unification_γ075_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {json_path}")
    
    # Generate summary report
    report = f"Einstein Unification Test Suite Summary ({timestamp})\n"
    report += "=============================================\n"
    report += f"Theory: Linear Signal Loss (γ={GAMMA})\n"
    report += f"Runs: {NUM_RUNS} | Device: {DEVICE} | Dtype: {DTYPE}\n\n"
    for test_name, data in summary['tests'].items():
        report += f"{test_name.upper()}:\n"
        report += f"  Mean Balance: {data['stats']['mean_balance']:.6f}\n"
        report += f"  Std Balance: {data['stats']['std_balance']:.6f}\n"
        report += f"  p-value (symmetry): {data['stats']['p_symmetry']:.4f}\n"
        report += f"  Interpretation: {data['stats']['interpretation']}\n"
        report += f"  Plot: {data['plot']}\n\n"
    report += "Q Sensitivity (Pulsar Anomaly Balance):\n"
    for item in q_sensitivity:
        report += f"  Q={item['Q']:.1e}: Balance={item['balance']:.6f}\n"
    report += "\n<reason>Summary: If mean_balance ≈0 and p>0.05 across tests, this supports unification at γ=0.75—gravity mimicking EM via information loss, echoing Einstein's non-symmetric metric dreams.</reason>"
    
    report_path = os.path.join(RESULTS_DIR, f'summary_γ075_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    main() 