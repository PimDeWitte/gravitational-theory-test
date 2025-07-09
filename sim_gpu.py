import torch
import time
from scipy.constants import G, c, k, hbar, epsilon_0
import numpy as np
import os

# --- Device Setup: The core of the GPU acceleration ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("✅ PyTorch MPS (Metal) device found. Running on GPU.")
else:
    device = torch.device("cpu")
    print("⚠️ PyTorch MPS device not found. Running on CPU.")

DTYPE = torch.float32
print(f"⚠️ Using low-precision {DTYPE} for GPU compatibility. Results are for performance testing, not final accuracy.")

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# B. SIMULATION PARAMETERS
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
M_val = 1.989e30 * 10
c_val = c
G_val = G
RS_val = (2 * G_val * M_val) / (c_val**2)

M = torch.tensor(M_val, device=device, dtype=DTYPE)
RS = torch.tensor(RS_val, device=device, dtype=DTYPE)
EPSILON = 1e-9 # Increased precision for stability

J_FRAC = 0.5; Q_PARAM = 1e12; Q_UNIFIED = 1e12; ASYMMETRY_PARAM = 1e-4
TORSION_PARAM = 1e-3; OBSERVER_ENERGY = 1e9; LAMBDA_COSMO = 1.11e-52

# --- MODIFICATION: Add flags for exploratory vs. final runs ---
FINAL_RUN = False # Set to True for the full, high-fidelity simulation for final results
VERBOSE_DEBUG = False # Set to True to print step-by-step trajectory data for GR

print("="*80)
print(f"PYTORCH-BASED GPU ORBITAL TEST | ALL THEORIES | {M.cpu().numpy()/1.989e30:.1f} M_sol BLACK HOLE")
print("="*80)

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# C. GRAVITATIONAL THEORY IMPLEMENTATIONS (PYTORCH FLOAT32 VERSION)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

class GravitationalTheory:
    def __init__(self, name): self.name = name
    def get_metric(self, r, M_param, C_param, G_param): raise NotImplementedError

class Schwarzschild(GravitationalTheory):
    def __init__(self): super().__init__("Schwarzschild (GR)")
    def get_metric(self, r, M_param, C_param, G_param):
        rs_local = (2 * G_param * M_param) / (C_param**2)
        # Add epsilon to denominator to prevent division by zero if r is exactly rs_local
        m = 1 - rs_local / (r + EPSILON)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class NewtonianLimit(GravitationalTheory):
    def __init__(self): super().__init__("Newtonian Limit")
    def get_metric(self, r, M_param, C_param, G_param):
        rs_local = (2 * G_param * M_param) / (C_param**2)
        return -(1 - rs_local / r), torch.ones_like(r), r**2, torch.zeros_like(r)

class ReissnerNordstrom(GravitationalTheory):
    def __init__(self, Q):
        super().__init__(f"Reissner-Nordström (Q={Q:.1e})")
        self.Q = torch.tensor(Q, device=device, dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs_local = (2 * G_param * M_param) / C_param**2
        r_q2 = (self.Q**2 * G_param) / (4 * torch.pi * epsilon_0 * C_param**4)
        m = 1 - rs_local / r + r_q2 / r**2
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class Kerr(GravitationalTheory):
    def __init__(self, J_frac):
        super().__init__(f"Kerr (a* = {J_frac:.2f})")
        self.a = J_frac * G_val * M_val / c_val
    def get_metric(self, r, M_param, C_param, G_param):
        rs_local = (2 * G_param * M_param) / C_param**2
        a2 = self.a**2
        rho2 = r**2
        delta = r**2 - rs_local * r + a2
        g_tt = -(1 - rs_local * r / rho2)
        g_rr = rho2 / (delta + EPSILON)
        g_pp = (r**2 + a2)**2 - delta * a2
        g_tp = -rs_local * self.a * r / rho2
        return g_tt, g_rr, g_pp, g_tp

class EinsteinFinalEquation(GravitationalTheory):
    def __init__(self, alpha):
        super().__init__(f"Einstein's Final Eq. (α={alpha:.2f})")
        self.alpha = torch.tensor(alpha, device=device, dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = (2 * G_param * M_param) / C_param**2
        base_metric = 1 - rs / r
        correction = self.alpha * (rs / r)**3
        final_metric = base_metric + correction
        return -final_metric, 1 / (final_metric + EPSILON), r**2, torch.zeros_like(r)

class EinsteinUnifiedFinal(GravitationalTheory):
    def __init__(self,q):
        super().__init__(f"Einstein's Unified (q={q:.1e})")
        self.q=torch.tensor(q,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs,r_q2=(2*G_param*M_param)/C_param**2,(self.q**2*G_param)/(4*torch.pi*epsilon_0*C_param**4)
        m=1-rs/r+r_q2/r**2
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class EinsteinAsymmetric(GravitationalTheory):
    def __init__(self,alpha):
        super().__init__(f"Einstein Asymmetric (α={alpha:.1e})")
        self.alpha=torch.tensor(alpha,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        m=1-rs/r+self.alpha*(rs/r)**2
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class EinsteinTeleparallel(GravitationalTheory):
    def __init__(self,tau):
        super().__init__(f"Einstein Teleparallel (τ={tau:.1e})")
        self.tau=torch.tensor(tau,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        base=1-rs/r
        corr=self.tau*(rs/r)**3
        return -base,1/(base-corr+EPSILON),r**2,torch.zeros_like(r)

class EinsteinRegularized(GravitationalTheory):
    def __init__(self): super().__init__("Einstein Regularized Core")
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        lp=torch.sqrt(torch.tensor(G_param*hbar/C_param**3,device=device,dtype=DTYPE))
        m=1-rs/torch.sqrt(r**2+lp**2)
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class Yukawa(GravitationalTheory):
    def __init__(self,lambda_mult):
        super().__init__(f"Yukawa (λ={lambda_mult:.2f}*RS)")
        self.lambda_mult=torch.tensor(lambda_mult,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        m=1-(rs/r)*torch.exp(-r/(self.lambda_mult*rs))
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class QuantumCorrected(GravitationalTheory):
    def __init__(self,alpha):
        super().__init__(f"Quantum Corrected (α={alpha:.2f})")
        self.alpha=torch.tensor(alpha,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        m=1-rs/r+self.alpha*(rs/r)**3
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class HigherDimensional(GravitationalTheory):
    def __init__(self,crossover_mult):
        super().__init__(f"Higher-Dim (cross={crossover_mult:.2f}*RS)")
        self.rc=torch.tensor(crossover_mult*RS_val,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        p4d,p5d=rs/r,(self.rc*rs)/r**2
        t=1/(1+torch.exp(-(r-self.rc)/(self.rc/10)))
        m=1-(t*p4d+(1-t)*p5d)
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class LogCorrected(GravitationalTheory):
    def __init__(self,beta):
        super().__init__(f"Log Corrected (β={beta:.2f})")
        self.beta=torch.tensor(beta,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        sr=torch.maximum(r,rs*1.001)
        lc=self.beta*(rs/sr)*torch.log(sr/rs)
        m=1-rs/r+lc
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class VariableG(GravitationalTheory):
    def __init__(self,delta):
        super().__init__(f"Variable G (δ={delta:.2f})")
        self.delta=torch.tensor(delta,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        G_eff=G_param*(1+self.delta*rs/r)
        m=1-(2*G_eff*M_param)/(C_param**2*r)
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class NonLocal(GravitationalTheory):
    def __init__(self): super().__init__("Non-local (Cosmological)")
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        ct=torch.tensor(LAMBDA_COSMO,device=device,dtype=DTYPE)*r**2/3
        m=1-rs/r-ct
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class Fractal(GravitationalTheory):
    def __init__(self,D):
        super().__init__(f"Fractal (D={D:.2f})")
        self.D=torch.tensor(D,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        m=1-(rs/r)**(self.D-2.0)
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class PhaseTransition(GravitationalTheory):
    def __init__(self,crit_mult):
        super().__init__(f"Phase Transition(crit={crit_mult:.2f}*RS)")
        self.rc=torch.tensor(crit_mult*RS_val,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        n=1-rs/r
        co=1-rs/self.rc
        m=torch.where(r>self.rc,n,co)
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class Acausal(GravitationalTheory):
    def __init__(self): super().__init__("Acausal (Final State)")
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        ht=(hbar*C_param**3)/(8*torch.pi*G_param*M_param*k)
        pt=torch.sqrt(torch.tensor(hbar*C_param**5/(G_param*k**2),device=device,dtype=DTYPE))
        cf=1-(ht/pt)
        m=1-(rs*cf)/r
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class Computational(GravitationalTheory):
    def __init__(self): super().__init__("Computational Complexity")
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        lp=torch.sqrt(torch.tensor(G_param*hbar/C_param**3,device=device,dtype=DTYPE))
        sr=torch.maximum(r,lp)
        cf=(rs**2)
        m=1-cf/(sr*torch.log2(sr/lp))
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class Tduality(GravitationalTheory):
    def __init__(self): super().__init__("String T-Duality")
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        re=r+rs**2/r # T-Duality length is the Schwarzschild radius
        m=1-rs/re
        return -m,1/(m+EPSILON),r**2,torch.zeros_like(r)

class Hydrodynamic(GravitationalTheory):
    def __init__(self): super().__init__("Emergent (Hydrodynamic)")
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        vfs=torch.minimum((rs*C_param**2)/r,torch.tensor(C_param**2*0.999999,device=device,dtype=DTYPE))
        gs=1/(1-vfs/C_param**2+EPSILON)
        return -1/gs,gs,r**2,torch.zeros_like(r)

class Participatory(GravitationalTheory):
    def __init__(self,obs_energy):
        super().__init__(f"Participatory(E_obs={obs_energy:.1e})")
        self.obs_energy=torch.tensor(obs_energy,device=device,dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs=(2*G_param*M_param)/C_param**2
        Ep=torch.sqrt(torch.tensor(hbar*C_param**5/G_param,device=device,dtype=DTYPE))
        cert=1-torch.exp(-5*self.obs_energy/Ep)
        g_tt_gr,g_rr_gr=-(1-rs/r),1/(1-rs/r+EPSILON)
        g_tt_vac,g_rr_vac=-1.0,1.0
        g_tt=cert*g_tt_gr+(1-cert)*g_tt_vac
        g_rr=cert*g_rr_gr+(1-cert)*g_rr_vac
        return g_tt,g_rr,r**2,torch.zeros_like(r)

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# D. NEW, ROBUST PHYSICS ENGINE
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

class GeodesicIntegrator:
    def __init__(self, model, y0_full, M_param, C_param, G_param):
        self.model = model
        self.M = M_param
        self.c = C_param
        self.G = G_param

        # Calculate conserved quantities (Energy and Angular Momentum) from initial conditions
        # These are constants of motion for the entire trajectory.
        _, r0, _, dt_dtau0, dr_dtau0, dphi_dtau0 = y0_full
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r0, self.M, self.c, self.G)
        
        # Using the definition of the 4-velocity u^μ = dx^μ/dτ and the metric,
        # the conserved energy E and angular momentum Lz per unit mass are:
        # E = -g_tμ u^μ = -(g_tt * c*dt/dτ + g_tφ * dφ/dτ)
        # Lz = g_φμ u^μ = (g_φt * c*dt/dτ + g_φφ * dφ/dτ)
        # Your code's dt_dtau is dimensionless dt/dτ, so u^t = c * dt_dtau
        self.E = -(g_tt * self.c * dt_dtau0 + g_tp * dphi_dtau0)
        self.Lz = g_tp * self.c * dt_dtau0 + g_pp * dphi_dtau0

    def ode_system(self, y_state):
        """
        Calculates the derivatives [dt/dτ, dr/dτ, dφ/dτ, d²r/dτ²].
        This is a system of first-order ODEs for [t, r, φ, dr/dτ].
        """
        _, r, _, dr_dtau = y_state

        # --- MODIFICATION: Calculate radial acceleration using the effective potential ---
        # The equation for radial motion can be derived from u_μ u^μ = -c^2
        # (dr/dτ)^2 = V_eff(r). We need d/dτ of this, which gives 2*r'*r'' = d(V_eff)/dr * r'
        # So, r'' = 0.5 * d(V_eff)/dr.
        # We use torch.autograd to get the derivative of the effective potential w.r.t r.
        r_grad = r.clone().detach().requires_grad_(True)
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r_grad, self.M, self.c, self.G)

        # The effective potential V_eff is derived from rearranging the normalization condition
        # g_rr (dr/dτ)^2 = -c^2 - g_tt(u^t)^2 - g_φφ(u^φ)^2 - 2*g_tφ*u^t*u^φ
        # We can solve for u^t and u^φ in terms of conserved E and Lz.
        det = g_tp**2 - g_tt * g_pp
        if torch.abs(det) < EPSILON:
             return torch.zeros(4, device=device, dtype=DTYPE)

        # From E = -g_tt*u^t - g_tφ*u^φ and Lz = g_φt*u^t + g_φφ*u^φ
        # We can invert to find u^t and u^φ (where u^t = c*dt/dτ, u^φ = dφ/dτ)
        u_t = (self.E * g_pp + self.Lz * g_tp) / det
        u_phi = -(self.E * g_tp + self.Lz * g_tt) / det
        
        # This is (dr/dτ)^2, the effective radial potential
        V_eff_sq = (-self.c**2 - (g_tt * u_t**2 + g_pp * u_phi**2 + 2 * g_tp * u_t * u_phi)) / g_rr
        
        # Calculate the gradient d(V_eff²)/dr to find the acceleration
        # r_grad.grad will be None if no operation that requires grad is performed.
        # Ensure backward() is called on a scalar.
        if V_eff_sq.dim() > 0:
             V_eff_sq.sum().backward()
        else:
            V_eff_sq.backward()

        d_Veff_sq_dr = r_grad.grad
        if d_Veff_sq_dr is None:
            d_Veff_sq_dr = torch.zeros_like(r)

        # d²r/dτ² = (1/2) * d(V_eff²)/dr
        d2r_dtau2 = 0.5 * d_Veff_sq_dr

        # Derivatives for the state vector [t, r, phi, dr/dtau]
        dt_dtau = u_t / self.c # Make it dimensionless for the state vector
        dphi_dtau = u_phi
        
        return torch.stack([dt_dtau, dr_dtau, dphi_dtau, d2r_dtau2])

    def rk4_step(self, y, dt):
        """
        Performs a single Runge-Kutta 4th order step.
        y is the state vector: [t, r, phi, dr/dtau]
        """
        k1 = self.ode_system(y)
        k2 = self.ode_system(y + 0.5 * dt * k1)
        k3 = self.ode_system(y + 0.5 * dt * k2)
        k4 = self.ode_system(y + dt * k3)
        
        # Update state vector
        y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0 * dt
        return y_new

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# E. MAIN SIMULATION SCRIPT
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
if __name__ == "__main__":
    total_start_time = time.time()

    models_to_test = [
        Schwarzschild(), NewtonianLimit(), Acausal(), Kerr(J_frac=J_FRAC),
        ReissnerNordstrom(Q=Q_PARAM), NonLocal(), Computational(), Tduality(),
        Hydrodynamic(), Participatory(obs_energy=OBSERVER_ENERGY), EinsteinUnifiedFinal(q=Q_UNIFIED),
        EinsteinAsymmetric(alpha=ASYMMETRY_PARAM), EinsteinTeleparallel(tau=TORSION_PARAM),
        EinsteinRegularized(),
    ]

    param_sweeps = {
        "Quantum Corrected": (QuantumCorrected, {"alpha": np.linspace(-2.0, 2.0, 10)}),
        "Log Corrected": (LogCorrected, {"beta": np.linspace(-1.5, 1.5, 10)}),
        "Yukawa": (Yukawa, {"lambda_mult": np.logspace(np.log10(1.5), 2, 10)}),
        "Variable G": (VariableG, {"delta": np.linspace(-0.5, 0.5, 10)}),
        "Einstein's Final Eq.": (EinsteinFinalEquation, {"alpha": np.linspace(-1.0, 1.0, 5)}),
        "Fractal": (Fractal, {"D": np.linspace(2.95, 3.05, 10)}),
        "Phase Transition": (PhaseTransition, {"crit_mult": np.array([1.5, 2.5, 4.0, 8.0, 16.0])}),
        "Higher-Dim": (HigherDimensional, {"crossover_mult": np.array([2.0, 10.0, 20.0, 50.0])})
    }
    for name_prefix, (model_class, params) in param_sweeps.items():
        param_key = list(params.keys())[0]
        for val in params[param_key]:
            models_to_test.append(model_class(**{param_key: val}))

    total_models = len(models_to_test)
    print(f"Initializing PyTorch GPU orbital test for {total_models} universes...\n")

    # --- Initial Conditions ---
    r0 = 4.0 * RS
    v_tan = torch.sqrt((G * M) / r0) # Classical circular orbit velocity
    
    # Use Schwarzschild metric to find initial 4-velocities for a quasi-stable orbit
    g_tt0, _, g_pp0, _ = Schwarzschild().get_metric(r0, M, c_val, G_val)
    
    # Normalization factor for 4-velocity, assuming initial dr/dτ = 0
    # u_μ u^μ = g_tt(u^t)^2 + g_φφ(u^φ)^2 = -c^2
    # And u^φ/u^t ≈ v_tan/r0. This gives the normalization for u^t and u^φ
    norm_factor_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * c_val))**2
    dt_dtau0 = 1.0 / torch.sqrt(norm_factor_sq)
    dphi_dtau0 = (v_tan / r0) * dt_dtau0

    # Full state vector: [t, r, phi, dt/dtau, dr/dtau, dphi/dtau]
    y0_full = torch.tensor([0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()], device=device, dtype=DTYPE)
    
    # The state vector for the integrator is [t, r, phi, dr/dtau]
    y0_integ = y0_full[[0, 1, 2, 4]]

    # --- MODIFICATION: Set number of steps based on run type ---
    if FINAL_RUN:
        print("🚀 Running in FINAL mode: Using high step count for precision.")
        N_STEPS = 250_000
        progress_interval = 25_000
    else:
        print("🧪 Running in EXPLORATORY mode: Using low step count for speed.")
        N_STEPS = 5_000
        progress_interval = 1_000
    
    DT = 0.2

    # --- Ground Truth Caching for GR ---
    GROUND_TRUTH_GR_FILE = f"ground_truth_gr_{N_STEPS}steps.pt"
    ground_truth_final_state = None

    if os.path.exists(GROUND_TRUTH_GR_FILE):
        print(f"✅ Loading GR ground truth from {GROUND_TRUTH_GR_FILE}...")
        ground_truth_final_state = torch.load(GROUND_TRUTH_GR_FILE, map_location=device)
    else:
        print(f"⏳ Running ground truth simulation with Schwarzschild (GR) for {N_STEPS:,} steps...")
        integrator_true = GeodesicIntegrator(models_to_test[0], y0_full, M, c_val, G_val)
        y_true = y0_integ.clone()
        
        start_time_gt = time.time()
        for step in range(N_STEPS):
            y_true = integrator_true.rk4_step(y_true, DT)
            if VERBOSE_DEBUG and (step+1) % 500 == 0:
                # Use .detach() here for safe printing
                y_print = y_true.detach()
                print(f"  GR Step {(step+1):,}: r={y_print[1]/RS:.6f}*RS, dr/dτ={y_print[3]:.4e}")
            if (step+1) % progress_interval == 0:
                elapsed = time.time() - start_time_gt
                print(f"  Step {(step+1):,}/{N_STEPS:,} ({(step+1)/N_STEPS*100:.1f}%) - r={y_true.detach()[1]/RS:.4f}*RS - Elapsed: {elapsed:.1f}s")
            if y_true[1] <= RS * 1.01: # Add a small buffer to stop before singularity
                print(f"  Event horizon reached at step {step+1:,}")
                break
        
        ground_truth_final_state = y_true
        torch.save(ground_truth_final_state, GROUND_TRUTH_GR_FILE)
        print(f"✅ GR ground truth simulation complete. Result cached to {GROUND_TRUTH_GR_FILE}.")

    # --- BUG FIX: Use .detach() before calling .numpy() ---
    ground_truth_final_state_np = ground_truth_final_state.cpu().detach().numpy()
    print(f"GR orbit loaded/completed. Final r={ground_truth_final_state_np[1]/RS.cpu().detach():.4f}*RS, φ={np.rad2deg(ground_truth_final_state_np[2]):.2f}°\n")

    # --- Ground Truth Caching for Kaluza-Klein ---
    GROUND_TRUTH_KK_FILE = f"ground_truth_kk_{N_STEPS}steps.pt"
    kaluza_klein_final_state = None
    
    if os.path.exists(GROUND_TRUTH_KK_FILE):
        print(f"✅ Loading Kaluza-Klein ground truth from {GROUND_TRUTH_KK_FILE}...")
        kaluza_klein_final_state = torch.load(GROUND_TRUTH_KK_FILE, map_location=device)
    else:
        print(f"⏳ Running ground truth simulation with Kaluza-Klein for {N_STEPS:,} steps...")
        kk_model = ReissnerNordstrom(Q=Q_PARAM)
        integrator_kk = GeodesicIntegrator(kk_model, y0_full, M, c_val, G_val)
        y_kk = y0_integ.clone()

        start_time_kk = time.time()
        for step in range(N_STEPS):
            y_kk = integrator_kk.rk4_step(y_kk, DT)
            if (step+1) % progress_interval == 0:
                elapsed = time.time() - start_time_kk
                print(f"  Step {(step+1):,}/{N_STEPS:,} ({(step+1)/N_STEPS*100:.1f}%) - r={y_kk.detach()[1]/RS:.4f}*RS - Elapsed: {elapsed:.1f}s")
            if y_kk[1] <= RS * 1.01:
                print(f"  Event horizon reached at step {step+1:,}")
                break
        
        kaluza_klein_final_state = y_kk
        torch.save(kaluza_klein_final_state, GROUND_TRUTH_KK_FILE)
        print(f"✅ Kaluza-Klein ground truth simulation complete. Result cached to {GROUND_TRUTH_KK_FILE}.")

    # --- BUG FIX: Use .detach() before calling .numpy() ---
    kaluza_klein_final_state_np = kaluza_klein_final_state.cpu().detach().numpy()
    print(f"KK orbit loaded/completed. Final r={kaluza_klein_final_state_np[1]/RS.cpu().detach():.4f}*RS, φ={np.rad2deg(kaluza_klein_final_state_np[2]):.2f}°\n")

    results = []
    print("--- Simulating Orbital Trajectories for All Models (PyTorch GPU) ---")
    for model in models_to_test:
        print(f"Testing: {model.name}...")
        model_start_time = time.time()
        
        integrator_pred = GeodesicIntegrator(model, y0_full, M, c_val, G_val)
        y_pred = y0_integ.clone()

        for i in range(N_STEPS):
            y_pred = integrator_pred.rk4_step(y_pred, DT)
            if y_pred[1] <= RS * 1.01:
                break

        # --- BUG FIX: Use .detach() before calling .numpy() ---
        final_state_pred = y_pred.cpu().detach().numpy()
        
        # Ground truth state is [t, r, phi, dr/dtau]. We only need r and phi for loss.
        r_true, phi_true = ground_truth_final_state_np[1], ground_truth_final_state_np[2]
        r_pred, phi_pred = final_state_pred[1], final_state_pred[2]

        # Correct Loss Calculation (Law of Cosines) for squared distance (m^2)
        loss_sq = (r_true**2 + r_pred**2 - 2 * r_true * r_pred * np.cos(phi_true - phi_pred))

        if not np.isfinite(loss_sq):
            loss_sq = 1e50 # Assign a large penalty for unstable/failed simulations

        results.append({"Model": model.name, "Loss": loss_sq})
        print(f"  -> Finished in {time.time()-model_start_time:.2f}s. Deviation: {loss_sq:.4e} m²")

    print("\n--- PYTORCH GPU ORBITAL TEST (FLOAT32): RANKED DEVIATION ---")
    results.sort(key=lambda x: x["Loss"])
    print(f"{'Rank':<5} | {'Model Name':<65} | {'Final Trajectory Deviation (m²)':<35}")
    print("="*110)
    for rank, res in enumerate(results, 1):
        loss_str = f"{res['Loss']:.4e}" if res['Loss'] < 1e49 else "Solver Failed or Plunged"
        print(f"{rank:<5} | {res['Model']:<65} | {loss_str}")
    print("="*110)
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")