import torch
import numpy as np
from self_discovery import run_trajectory, c, G, G_T, device, DTYPE, TORCH_PI  # Import necessary from main script
from linear_signal_loss import LinearSignalLoss_gamma_1_00, QuantumLinearSignalLoss

# Real observational data (PSR B1913+16)
# Measured periastron advance: 4.226595 deg/yr (Weisberg & Huang 2016)
# For demo, we'll use this value; in practice, load from dataset
REAL_ADVANCE = 4.226595  # deg/yr

# System parameters for PSR B1913+16 (approximate for simulation)
# Note: For accurate comparison, scale to match system masses/orbits
PERIOD_DAYS = 0.322997448918  # Orbital period in days
PERIOD = PERIOD_DAYS * 86400  # seconds
M_SI = 1.4 * 1.989e30  # Approximate pulsar mass in kg
RS_SI = 2 * G * M_SI / c**2
M = torch.as_tensor(M_SI, device=device, dtype=DTYPE)
RS = torch.as_tensor(RS_SI, device=device, dtype=DTYPE)


def compute_periastron_advance(traj: torch.Tensor) -> float:
    """Compute periastron advance in deg/yr from trajectory.
    Assumes traj[:,2] is phi (azimuthal angle).
    """
    phi = traj[:, 2].cpu().numpy()
    delta_phi = phi[-1] - phi[0]  # Total angle change
    r0 = traj[0,1]
    v_tan = torch.sqrt(G_T * M / r0)
    period_est = 2 * TORCH_PI * r0 / v_tan
    dtau = period_est / 1000.0  # Match main()
    num_orbits = len(traj) * dtau.item() / PERIOD  # Total time / period
    if num_orbits == 0: return 0.0
    advance_rad = delta_phi - 2 * np.pi * num_orbits  # Excess angle
    advance_deg = np.degrees(float(advance_rad)) / num_orbits * (365.25 / PERIOD_DAYS)  # deg/yr
    return advance_deg

def validate_against_observations(model, r0: torch.Tensor, N_STEPS: int, max_failures: int, step_print: int) -> dict:
    v_tan = torch.sqrt(G_T * M / r0)
    period_est = 2 * TORCH_PI * r0 / v_tan
    DTau = period_est / 1000.0
    traj, tag = run_trajectory(model, r0, N_STEPS, DTau, max_failures, step_print)
    simulated_advance = compute_periastron_advance(traj)
    loss = abs(simulated_advance - REAL_ADVANCE)
    return {'model': model.name, 'sim_advance': simulated_advance, 'loss': loss}

# Example usage
if __name__ == '__main__':
    r0 = torch.tensor(1e9, device=device, dtype=DTYPE)  # Approximate semi-major axis in meters (~1 million km)
    N_STEPS = 100000  # Validation mode
    model = LinearSignalLoss_gamma_1_00()
    results = validate_against_observations(model, r0, N_STEPS, 10, N_STEPS//50)
    print(f"{results['model']}: Simulated Advance = {results['sim_advance']:.6f} deg/yr, Loss = {results['loss']:.6f}")
    
    quantum_model = QuantumLinearSignalLoss(beta=0.1)
    q_results = validate_against_observations(quantum_model, r0, N_STEPS, 10, N_STEPS//50)
    print(f"{q_results['model']}: Simulated Advance = {q_results['sim_advance']:.6f} deg/yr, Loss = {q_results['loss']:.6f}") 