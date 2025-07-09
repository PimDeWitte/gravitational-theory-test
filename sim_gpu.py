#!/usr/bin/env python3
# sim_gpu.py  ── July 2025
# ---------------------------------------------------------------------------
# Float‑32 black‑hole orbital integrator for Apple‑silicon (M‑series) GPUs.
# All known mathematical / computational bugs are fixed; optional Torch‑Dynamo
# compilation can be enabled with `TORCH_COMPILE=1`.
# ---------------------------------------------------------------------------

from __future__ import annotations
import os, time, math, argparse, warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c, k, hbar, epsilon_0

# ---------------------------------------------------------------------------
# 0.  DEVICE & GLOBAL CONSTANTS
# ---------------------------------------------------------------------------

DTYPE  = torch.float32
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TORCH_PI = torch.as_tensor(math.pi,  device=device, dtype=DTYPE)
EPS0_T   = torch.as_tensor(epsilon_0, device=device, dtype=DTYPE)
EPSILON  = torch.finfo(DTYPE).eps * 100         # ~1e‑5 for float32

# 10 M☉ black hole -----------------------------------------------------------
M_SI  = 10.0 * 1.989e30
RS_SI = 2 * G * M_SI / c**2
M  = torch.as_tensor(M_SI , device=device, dtype=DTYPE)
RS = torch.as_tensor(RS_SI, device=device, dtype=DTYPE)

# Planck length (cached tensor)
LP = torch.as_tensor(math.sqrt(G * hbar / c**3), device=device, dtype=DTYPE)

# Model parameters -----------------------------------------------------------
# --- FIX: Increase Q_PARAM for a physically significant Reissner-Nordström metric ---
J_FRAC, Q_PARAM, Q_UNIFIED         = 0.5, 3.0e14, 1.0e12
ASYMMETRY_PARAM, TORSION_PARAM     = 1.0e-4, 1.0e-3
OBSERVER_ENERGY, LAMBDA_COSMO      = 1.0e9, 1.11e-52

# ---------------------------------------------------------------------------
# 1.  BASE CLASS & METRIC DEFINITIONS
# ---------------------------------------------------------------------------

Tensor = torch.Tensor  # typing alias


class GravitationalTheory:
    name: str

    def __init__(self, name: str) -> None: self.name = name

    def get_metric(
        self, r: Tensor, M_param: Tensor | float, C_param: float, G_param: float
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError


# -- 1.1 Standard metrics ----------------------------------------------------


class Schwarzschild(GravitationalTheory):
    def __init__(self): super().__init__("Schwarzschild (GR)")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / (r + EPSILON)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class NewtonianLimit(GravitationalTheory):
    def __init__(self): super().__init__("Newtonian Limit")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        return -m, torch.ones_like(r), r**2, torch.zeros_like(r)


class ReissnerNordstrom(GravitationalTheory):
    def __init__(self, Q: float):
        super().__init__(f"Reissner‑Nordström (Q={Q:.1e})")
        self.Q = torch.as_tensor(Q, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4 * r**2)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class Kerr(GravitationalTheory):
    def __init__(self, J_frac: float):
        super().__init__(f"Kerr (a*={J_frac:.3f})")
        self.J_frac = float(J_frac)

    def get_metric(self, r, M_param, C_param, G_param):
        a = torch.as_tensor(self.J_frac * G_param * M_param / C_param,
                            device=r.device, dtype=r.dtype)
        rs  = 2 * G_param * M_param / C_param**2
        rho2 = r**2
        Δ = r**2 - rs * r + a**2

        g_tt = -(1 - rs * r / rho2)
        g_rr = rho2 / (Δ + EPSILON)
        g_pp = ((r**2 + a**2) ** 2 - Δ * a**2) / rho2
        g_tp = -rs * a * r / rho2
        return g_tt, g_rr, g_pp, g_tp


# -- 1.2 Selected speculative metrics (all previously fixed versions) --------

class EinsteinFinalEquation(GravitationalTheory):
    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + self.alpha * (rs / r) ** 3
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class EinsteinUnifiedFinal(GravitationalTheory):
    def __init__(self, q: float):
        super().__init__(f"Einstein Unified (q={q:.1e})")
        self.q = torch.as_tensor(q, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + (G_param * self.q**2) / (4 * TORCH_PI * EPS0_T * C_param**4 * r**2)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class EinsteinAsymmetric(GravitationalTheory):
    def __init__(self, alpha: float):
        super().__init__(f"Einstein Asymmetric (α={alpha:+.1e})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + self.alpha * (rs / r) ** 2
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class EinsteinTeleparallel(GravitationalTheory):
    def __init__(self, tau: float):
        super().__init__(f"Einstein Teleparallel (τ={tau:.1e})")
        self.tau = torch.as_tensor(tau, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r - self.tau * (rs / r) ** 3
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class EinsteinRegularized(GravitationalTheory):
    def __init__(self): super().__init__("Einstein Regularised Core")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / torch.sqrt(r**2 + LP**2)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class Yukawa(GravitationalTheory):
    def __init__(self, lambda_mult: float):
        super().__init__(f"Yukawa (λ={lambda_mult:.2f} RS)")
        self.lambda_mult = torch.as_tensor(lambda_mult, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs / r) * torch.exp(-r / (self.lambda_mult * rs))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class QuantumCorrected(GravitationalTheory):
    def __init__(self, alpha: float):
        super().__init__(f"Quantum Corrected (α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + self.alpha * (rs / r) ** 3
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class HigherDimensional(GravitationalTheory):
    def __init__(self, crossover_mult: float):
        super().__init__(f"Higher‑Dim (cross={crossover_mult:.1f} RS)")
        self.rc = torch.as_tensor(crossover_mult * RS_SI, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        p4d = rs / r
        p5d = (self.rc * rs) / r**2
        t = 1 / (1 + torch.exp(-(r - self.rc) / (self.rc / 10)))
        m = 1 - (t * p4d + (1 - t) * p5d)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class LogCorrected(GravitationalTheory):
    def __init__(self, beta: float):
        super().__init__(f"Log Corrected (β={beta:+.2f})")
        self.beta = torch.as_tensor(beta, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        sr = torch.maximum(r, rs * 1.001)
        lc = self.beta * (rs / sr) * torch.log(sr / rs)
        m = 1 - rs / r + lc
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class VariableG(GravitationalTheory):
    def __init__(self, delta: float):
        super().__init__(f"Variable G (δ={delta:+.2f})")
        self.delta = torch.as_tensor(delta, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        G_eff = G_param * (1 + self.delta * torch.log1p(r / rs))
        m = 1 - 2 * G_eff * M_param / (C_param**2 * r)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class NonLocal(GravitationalTheory):
    def __init__(self): super().__init__("Non‑local (Λ)")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        ct = torch.as_tensor(LAMBDA_COSMO, device=device, dtype=DTYPE) * r**2 / 3
        m = 1 - rs / r - ct
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class Fractal(GravitationalTheory):
    def __init__(self, D: float):
        super().__init__(f"Fractal (D={D:.3f})")
        self.D = torch.as_tensor(D, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs / r) ** (self.D - 2.0)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class PhaseTransition(GravitationalTheory):
    def __init__(self, crit_mult: float):
        super().__init__(f"Phase Transition (r_c={crit_mult:.2f} RS)")
        self.rc = torch.as_tensor(crit_mult * RS_SI, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m_out = 1 - rs / r
        m_in = 1 - rs / self.rc
        m = torch.where(r > self.rc, m_out, m_in)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class Acausal(GravitationalTheory):
    def __init__(self): super().__init__("Acausal (final‑state)")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        ht = hbar * C_param**3 / (8 * math.pi * G_param * M_param * k)
        pt = math.sqrt(hbar * C_param**5 / (G_param * k**2))
        cf = 1 - ht / pt
        m = 1 - (rs * cf) / r
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class Computational(GravitationalTheory):
    def __init__(self): super().__init__("Computational Complexity")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        sr = torch.maximum(r, LP)
        m = 1 - rs**2 / (sr * torch.log2(sr / LP))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class Tduality(GravitationalTheory):
    def __init__(self): super().__init__("T‑Duality (string)")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        re = r + rs**2 / r
        m = 1 - rs / re
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class Hydrodynamic(GravitationalTheory):
    def __init__(self): super().__init__("Emergent (hydrodynamic)")

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        v_cap = torch.as_tensor(0.999999 * C_param, device=r.device, dtype=r.dtype)
        vfs  = torch.minimum(rs / r * C_param, v_cap)
        gamma_sq = 1.0 / (1.0 - (vfs / C_param) ** 2 + EPSILON)
        return -gamma_sq, gamma_sq, r**2, torch.zeros_like(r)


class Participatory(GravitationalTheory):
    def __init__(self, obs_energy: float):
        super().__init__(f"Participatory (E_obs={obs_energy:.1e})")
        self.obs_energy = torch.as_tensor(obs_energy, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        Ep = math.sqrt(hbar * C_param**5 / G_param)
        cert = 1 - torch.exp(-5 * self.obs_energy / Ep)
        g_tt_gr = -(1 - rs / r)
        g_rr_gr = 1 / (1 - rs / r + EPSILON)
        g_tt = cert * g_tt_gr + (1 - cert) * (-1.0)
        g_rr = cert * g_rr_gr + (1 - cert) * 1.0
        return g_tt, g_rr, r**2, torch.zeros_like(r)


# ---------------------------------------------------------------------------
# 2.  GEODESIC INTEGRATOR (RK‑4)
# ---------------------------------------------------------------------------

class GeodesicIntegrator:
    """State vector: [t, r, φ, dr/dτ] (equatorial plane)"""

    def __init__(self, model: GravitationalTheory,
                 y0_full: Tensor, M_param: Tensor, C_param: float, G_param: float):
        self.model, self.M, self.c, self.G = model, M_param, C_param, G_param

        _, r0, _, dt_dtau0, _, dphi_dtau0 = y0_full
        g_tt0, _, g_pp0, g_tp0 = self.model.get_metric(r0, self.M, self.c, self.G)
        self.E  = -(g_tt0 * self.c * dt_dtau0 + g_tp0 * dphi_dtau0)
        self.Lz =  g_tp0 * self.c * dt_dtau0 + g_pp0 * dphi_dtau0

        # Optional Torch‑Dynamo compilation
        if os.environ.get("TORCH_COMPILE") == "1" and hasattr(torch, "compile"):
            try:
                self._ode = torch.compile(self._ode_impl,
                                          fullgraph=True,
                                          mode="reduce-overhead",
                                          dynamic=True)
            except Exception as exc:
                warnings.warn(f"torch.compile disabled: {exc}")
                self._ode = self._ode_impl
        else:
            self._ode = self._ode_impl

    # -----------------------------------------------------------------------
    # 2.1  RHS of the geodesic equations (needs autograd → DO NOT decorate
    #      with @torch.inference_mode)
    # -----------------------------------------------------------------------
    def _ode_impl(self, y_state: Tensor) -> Tensor:
        _, r, _, dr_dtau = y_state

        r_grad = r.clone().detach().requires_grad_(True)
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r_grad, self.M, self.c, self.G)

        det = g_tp ** 2 - g_tt * g_pp
        if torch.abs(det) < EPSILON:
            return torch.zeros_like(y_state)

        u_t   = (self.E * g_pp + self.Lz * g_tp) / det
        u_phi = -(self.E * g_tp + self.Lz * g_tt) / det

        V_sq = (-self.c ** 2 - (g_tt * u_t ** 2 + g_pp * u_phi ** 2
                                + 2 * g_tp * u_t * u_phi)) / g_rr
        (dV_dr,) = torch.autograd.grad(V_sq, r_grad,
                                       create_graph=False, retain_graph=False)
        d2r_dtau2 = 0.5 * dV_dr

        dt_dtau   = u_t / self.c
        dphi_dtau = u_phi

        return torch.stack((dt_dtau, dr_dtau, dphi_dtau, d2r_dtau2))

    # -----------------------------------------------------------------------
    # 2.2  Four‑stage RK‑4 integrator (detaches outputs to keep memory flat)
    # -----------------------------------------------------------------------
    def rk4_step(self, y: Tensor, dτ: float) -> Tensor:
        k1 = self._ode(y).detach()
        k2 = self._ode((y + 0.5 * dτ * k1)).detach()
        k3 = self._ode((y + 0.5 * dτ * k2)).detach()
        k4 = self._ode((y + dτ * k3)).detach()
        return y + (k1 + 2 * k2 + 2 * k3 + k4) * (dτ / 6.0)


# ---------------------------------------------------------------------------
# 3.  MAIN DRIVER
# ---------------------------------------------------------------------------

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--final", action="store_true", help="High‑precision run")
    p.add_argument("--plots", action="store_true", help="Plot every model")
    p.add_argument("--no-plots", action="store_true", help="Disable plots")
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    print("=" * 80)
    print(f"PyTorch‑MPS ORBITAL TEST | device={device} | dtype={DTYPE}")
    print("=" * 80)

    # -- 3.1  Model registry -------------------------------------------------
    models: list[GravitationalTheory] = [
        Schwarzschild(), NewtonianLimit(), Acausal(), Kerr(J_FRAC),
        ReissnerNordstrom(Q_PARAM), NonLocal(), Computational(), Tduality(),
        Hydrodynamic(), Participatory(OBSERVER_ENERGY),
        EinsteinUnifiedFinal(Q_UNIFIED), EinsteinAsymmetric(ASYMMETRY_PARAM),
        EinsteinTeleparallel(TORSION_PARAM), EinsteinRegularized(),
    ]

    sweeps = {
        "QuantumCorrected": (QuantumCorrected,
                             dict(alpha=np.linspace(-2.0, 2.0, 10))),
        "LogCorrected": (LogCorrected,
                         dict(beta=np.linspace(-1.5, 1.5, 10))),
        "Yukawa": (Yukawa,
                   dict(lambda_mult=np.logspace(math.log10(1.5), 2, 10))),
        "VariableG": (VariableG,
                      dict(delta=np.linspace(-0.5, 0.5, 10))),
        "EinsteinFinal": (EinsteinFinalEquation,
                          dict(alpha=np.linspace(-1.0, 1.0, 5))),
        "Fractal": (Fractal,
                    dict(D=np.linspace(2.95, 3.05, 10))),
        "PhaseTransition": (PhaseTransition,
                            dict(crit_mult=np.array([1.5, 2.5, 4.0, 8.0, 16.0]))),
        "HigherDim": (HigherDimensional,
                      dict(crossover_mult=np.array([2.0, 10.0, 20.0, 50.0]))),
    }
    for cls, pd in sweeps.values():
        key, vals = next(iter(pd.items()))
        models += [cls(**{key: float(v)}) for v in vals]

    print(f"Total models: {len(models)}")

    # -- 3.2  Initial conditions --------------------------------------------
    r0 = 4.0 * RS
    v_tan = torch.sqrt(G * M / r0)
    g_tt0, _, g_pp0, _ = Schwarzschild().get_metric(r0, M, c, G)
    norm_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * c)) ** 2
    dt_dtau0 = 1.0 / torch.sqrt(norm_sq)
    dphi_dtau0 = (v_tan / r0) * dt_dtau0

    y0_full = torch.tensor(
        [0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()],
        device=device, dtype=DTYPE,
    )
    y0_state = y0_full[[0, 1, 2, 4]].clone()

    # -- 3.3  Run parameters -------------------------------------------------
    # --- FIX: Decrease DTau and increase N_STEPS for simulation stability ---
    DTau = 0.01
    if args.final:
        N_STEPS, STEP_PRINT, SAVE_PLOTS = 5_000_000, 250_000, True
        print("Mode: FINAL (high precision)")
    else:
        N_STEPS, STEP_PRINT, SAVE_PLOTS = 100_000, 10_000, args.plots
        print("Mode: EXPLORATORY (fast)")
    
    PLOT_DIR = "plots"; os.makedirs(PLOT_DIR, exist_ok=True)

    # -- 3.4  Ground‑truth trajectories (cached) ----------------------------
    def cached_run(model: GravitationalTheory, tag: str):
        fname = f"cache_{tag}_{N_STEPS}.pt"
        if os.path.exists(fname):
            return torch.load(fname, map_location=device)
        integ = GeodesicIntegrator(model, y0_full, M, c, G)
        hist = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
        hist[0] = y0_state
        y = y0_state.clone()
        for i in range(N_STEPS):
            y = integ.rk4_step(y, DTau)
            hist[i + 1] = y
            if (i + 1) % STEP_PRINT == 0:
                print(f"  {tag} step {i+1:,}/{N_STEPS:,} "
                      f"r={y[1]/RS:.3f} RS")
            if y[1] <= RS * 1.01:
                hist = hist[: i + 2]
                break
        torch.save(hist, fname)
        return hist

    GR_hist = cached_run(Schwarzschild(), "GR")
    RN_hist = cached_run(ReissnerNordstrom(Q_PARAM), "RN")
    GR_final, RN_final = GR_hist[-1], RN_hist[-1]

    # -- 3.5  Evaluate all models ------------------------------------------
    results = []
    for idx, model in enumerate(models, 1):
        print(f"[{idx:03}/{len(models)}] {model.name}")
        integ = GeodesicIntegrator(model, y0_full, M, c, G)
        traj = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
        traj[0] = y0_state
        y = y0_state.clone()
        for i in range(N_STEPS):
            y = integ.rk4_step(y, DTau)
            traj[i + 1] = y
            if y[1] <= RS * 1.01:
                traj = traj[: i + 2]
                break

        r_pred, phi_pred = y[1], y[2]

        def dev(final_ref):
            r_ref, phi_ref = final_ref[1], final_ref[2]
            if not torch.isfinite(r_pred) or not torch.isfinite(phi_pred):
                return float("inf")
            return (r_ref**2 + r_pred**2
                    - 2 * r_ref * r_pred * torch.cos(phi_ref - phi_pred)).item()

        results.append(dict(
            name=model.name,
            loss_GR=dev(GR_final),
            loss_RN=dev(RN_final),
            dot_GR=torch.dot(y, GR_final).item(),
            traj=traj.cpu().numpy(),
        ))

    results.sort(key=lambda d: d["loss_GR"])

    # -- 3.6  Text report ----------------------------------------------------
    print("\nRank | Model                               | Loss_GR")
    print("-"*60)
    for rank, res in enumerate(results, 1):
        print(f"{rank:4d} | {res['name']:<35} | {res['loss_GR']:10.3e}")

    # -- 3.7  Plots ----------------------------------------------------------
    if SAVE_PLOTS and not args.no_plots:
        GR_np, RN_np = GR_hist.cpu().numpy(), RN_hist.cpu().numpy()
        top = results if args.plots else results[:5]
        for res in top:
            pred_np = res["traj"]
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection="polar")
            ax.plot(GR_np[:, 2], GR_np[:, 1], "k--", label="GR")
            ax.plot(RN_np[:, 2], RN_np[:, 1], "b:",  label="R‑N")
            ax.plot(pred_np[:, 2], pred_np[:, 1], "r",  label=res["name"])
            ax.plot(pred_np[0, 2], pred_np[0, 1], "go", label="start")
            ax.plot(pred_np[-1, 2], pred_np[-1, 1], "rx", label="end")
            ax.set_title(res["name"], pad=20); ax.legend(); plt.tight_layout()
            safe = res["name"].translate({ord(c): "_" for c in " /()*"})
            plt.savefig(os.path.join(PLOT_DIR, f"{safe}.png"))
            plt.close()
        print(f"\nPlots saved to '{PLOT_DIR}/'.")

    print("Done.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()