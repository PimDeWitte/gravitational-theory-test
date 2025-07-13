#!/usr/bin/env python3
"""
Geodesic integrator for gravitational theories.
Extracted from self_discovery.py to avoid argument parsing conflicts.
"""

import torch
import os
from base_theory import GravitationalTheory, Tensor

class GeodesicIntegrator:
    """
    Integrates the geodesic equations for a given gravitational theory using RK4.
    """
    def __init__(self, model: GravitationalTheory, y0_full: Tensor, M_param: Tensor, C_param: float, G_param: float):
        """Initializes the integrator with a model and initial conditions."""
        self.model = model
        self.device = y0_full.device
        self.dtype = y0_full.dtype
        self.M = M_param.to(self.device, self.dtype) if isinstance(M_param, Tensor) else torch.tensor(M_param, device=self.device, dtype=self.dtype)
        self.c = C_param.to(self.device, self.dtype) if isinstance(C_param, Tensor) else torch.tensor(C_param, device=self.device, dtype=self.dtype)
        self.G = G_param.to(self.device, self.dtype) if isinstance(G_param, Tensor) else torch.tensor(G_param, device=self.device, dtype=self.dtype)
        _, r0, _, dt_dtau0, _, dphi_dtau0 = y0_full
        g_tt0, _, g_pp0, g_tp0 = self.model.get_metric(r0, self.M, self.c, self.G)
        self.E  = -(g_tt0 * self.c * dt_dtau0 + g_tp0 * dphi_dtau0)
        self.Lz =  g_tp0 * self.c * dt_dtau0 + g_pp0 * dphi_dtau0
        self.torsion_detected = False
        
        # Try to get EPSILON from environment
        try:
            dtype = self.dtype
            self.EPSILON = torch.finfo(dtype).eps * 100
        except:
            self.EPSILON = 1e-10
            
        if os.environ.get("TORCH_COMPILE") == "1" and hasattr(torch, "compile"):
            self._ode = torch.compile(self._ode_impl, fullgraph=True, mode="reduce-overhead", dynamic=True)
        else:
            self._ode = self._ode_impl

    def _ode_impl(self, y_state: Tensor) -> Tensor:
        """The right-hand side of the system of ODEs for the geodesic equations."""
        _, r, _, dr_dtau = y_state
        r_grad = r.clone().detach().requires_grad_(True)
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r_grad, self.M, self.c, self.G)
        if torch.any(g_tp != 0) and not self.torsion_detected:
            g_tp_mean = g_tp.mean().item()
            # Only print if torsion is significant
            if abs(g_tp_mean) > 1e-6:
                print(f"Torsion detected in {self.model.name}: g_tp mean = {g_tp_mean}")
            self.torsion_detected = True
        det = g_tp ** 2 - g_tt * g_pp
        if torch.abs(det) < self.EPSILON: return torch.zeros_like(y_state)
        u_t   = (self.E * g_pp + self.Lz * g_tp) / det
        u_phi = -(self.E * g_tp + self.Lz * g_tt) / det
        V_sq = (-self.c ** 2 - (g_tt * u_t ** 2 + g_pp * u_phi ** 2 + 2 * g_tp * u_t * u_phi)) / g_rr
        if not torch.all(torch.isfinite(V_sq)): return torch.full_like(y_state, float('nan'))
        (dV_dr,) = torch.autograd.grad(V_sq, r_grad, create_graph=False, retain_graph=False)
        d2r_dtau2 = 0.5 * dV_dr
        # Ensure all components are on the same device as the input
        # Convert scalar c to match device of tensors
        c_tensor = self.c.clone().detach().to(device=r.device, dtype=r.dtype)
        ut_comp = (u_t / c_tensor).to(r.device)
        dr_comp = dr_dtau.to(r.device)
        uphi_comp = u_phi.to(r.device)
        d2r_comp = d2r_dtau2.to(r.device)
        return torch.stack((ut_comp, dr_comp, uphi_comp, d2r_comp))

    def rk4_step(self, y: Tensor, dτ: float) -> Tensor:
        """Performs a single Runge-Kutta 4th order integration step."""
        k1 = self._ode(y).detach()
        k2 = self._ode((y + 0.5 * dτ * k1)).detach()
        k3 = self._ode((y + 0.5 * dτ * k2)).detach()
        k4 = self._ode((y + dτ * k3)).detach()
        return y + (k1 + 2 * k2 + 2 * k3 + k4) * (dτ / 6.0) 