# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# A. LIBRARY IMPORTS AND SETUP
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
import torch
import time
from scipy.constants import G, c, k, hbar, epsilon_0
import numpy as np
import os

DEBUG_COMPUTE = os.environ.get("DEBUG_COMPUTE", "0") == "1"

def debug_print(msg):
    if DEBUG_COMPUTE:
        print(msg)

def log(msg):
    # Only print log messages that are not routine, i.e., only for major events or compute-heavy steps
    print(f"[LOG] {msg}")

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
# B. SIMULATION PARAMETERS (as PyTorch Tensors on the GPU)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
M = torch.tensor(1.989e30 * 10, device=device, dtype=DTYPE)
RS = (2 * G * M) / (c**2)
EPSILON = 1e-7

J_FRAC = 0.5; Q_PARAM = 1e12; Q_UNIFIED = 1e12; ASYMMETRY_PARAM = 1e-4
TORSION_PARAM = 1e-3; OBSERVER_ENERGY = 1e9; LAMBDA_COSMO = 1.11e-52

print("="*80)
print(f"PYTORCH-BASED GPU ORBITAL TEST | ALL THEORIES | {M.cpu().numpy()/1.989e30:.1f} M_sol BLACK HOLE")
print("="*80)

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# C. PYTORCH-BASED PHYSICS ENGINE
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

def get_metric_derivatives_torch(model, r, M_const, C_const, G_const, h=1e-4):
    p_g = model.get_metric(r + h, M_const, C_const, G_const)
    m_g = model.get_metric(r - h, M_const, C_const, G_const)
    result = tuple([(p - m) / (2 * h) for p, m in zip(p_g, m_g)])
    return result

def geodesic_ode_pytorch(y, model, M_const, C_const, G_const):
    r, dt, dr, dphi = y[1], y[3], y[4], y[5]
    g_tt, g_rr, g_pp, g_tp = model.get_metric(r, M_const, C_const, G_const)
    g_ttp, g_rrp, g_ppp, g_tpp = get_metric_derivatives_torch(model, r, M_const, C_const, G_const)
    det_inv = g_tt * g_pp + g_tp**2
    if torch.abs(det_inv) < EPSILON: 
        return torch.zeros_like(y)
    d2r = -1/(2*g_rr) * (g_ttp*dt**2 + g_rrp*dr**2 + g_ppp*dphi**2 + 2*g_tpp*dt*dphi)
    d2t = (1/det_inv) * ((g_tp*g_ppp - g_tpp*g_pp)*dr*dphi + (g_tp*g_tpp - g_ttp*g_pp)*dr*dt)
    d2phi = (1/det_inv) * ((g_tpp*g_tt + g_ttp*g_tp)*dr*dphi + (g_ttp*g_pp + g_tp*g_tpp)*dr*dt)
    result = torch.stack([dt, dr, dphi, d2t, d2r, d2phi])
    return result

def rk4_step(y, func, model, dt):
    # Only print for compute-heavy steps if DEBUG_COMPUTE is enabled
    if DEBUG_COMPUTE:
        print(f"[LOG] Starting RK4 step for model {model.__class__.__name__} with dt={dt}...")
    k1 = func(y, model, M, c, G)
    k2 = func(y + 0.5*dt*k1, model, M, c, G)
    k3 = func(y + 0.5*dt*k2, model, M, c, G)
    k4 = func(y + dt*k3, model, M, c, G)
    result = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    if DEBUG_COMPUTE:
        print(f"[LOG]   k1: {k1}")
        print(f"[LOG]   k2: {k2}")
        print(f"[LOG]   k3: {k3}")
        print(f"[LOG]   k4: {k4}")
        print(f"[LOG]   RK4 result: {result}")
    return result

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# D. GRAVITATIONAL THEORY IMPLEMENTATIONS (PYTORCH FLOAT32 VERSION)
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-

class GravitationalTheory:
    def __init__(self, name): self.name = name
    def get_metric(self, r, M, C, G): raise NotImplementedError

class Schwarzschild(GravitationalTheory):
    def __init__(self): super().__init__("Schwarzschild (GR)")
    def get_metric(self, r, M, C, G):
        m=1-(2*G*M)/(C**2*r)
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class NewtonianLimit(GravitationalTheory):
    def __init__(self): super().__init__("Newtonian Limit")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/(C**2)
        result = (-(1-rs/r),torch.tensor(1.0,device=device,dtype=DTYPE),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class ReissnerNordstrom(GravitationalTheory):
    def __init__(self,Q): super().__init__(f"Reissner-Nordström (Q={Q:.1e})"); self.Q=torch.tensor(Q,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        r_q2=(self.Q**2*G)/(4*torch.pi*epsilon_0*C**4)
        m=1-rs/r+r_q2/r**2
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Kerr(GravitationalTheory):
    def __init__(self, J_frac): super().__init__(f"Kerr (a* = {J_frac:.2f})"); self.a=torch.tensor(J_frac*G*M.item()/c**2,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs,a2,rho2=(2*G*M)/C**2,self.a**2,r**2
        delta=rho2-rs*r+a2
        g_tt=-(1-rs*r/rho2)
        g_rr=rho2/(delta+EPSILON)
        g_pp=(r**2+a2)**2-delta*a2
        g_tp=-(rs*self.a*r/rho2)*C
        result = (g_tt/C**2,g_rr,g_pp,g_tp/C)
        return result

class EinsteinFinalEquation(GravitationalTheory):
    def __init__(self, alpha):
        super().__init__(f"Einstein's Final Eq. (α={alpha:.2f})")
        self.alpha = torch.tensor(alpha, device=device, dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs = (2 * G * M) / C**2
        base_metric = 1 - rs / r
        correction = self.alpha * (rs / r)**3
        final_metric = base_metric + correction
        return -final_metric, 1 / (final_metric + EPSILON), r**2, torch.tensor(0.0, device=device, dtype=DTYPE)

class EinsteinUnifiedFinal(GravitationalTheory):
    def __init__(self,q): super().__init__(f"Einstein's Unified (q={q:.1e})"); self.q=torch.tensor(q,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs,r_q2=(2*G*M)/C**2,(self.q**2*G)/(4*torch.pi*epsilon_0*C**4)
        m=1-rs/r+r_q2/r**2
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class EinsteinAsymmetric(GravitationalTheory):
    def __init__(self,alpha): super().__init__(f"Einstein Asymmetric (α={alpha:.1e})"); self.alpha=torch.tensor(alpha,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        m=1-rs/r+self.alpha*(rs/r)**2
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class EinsteinTeleparallel(GravitationalTheory):
    def __init__(self,tau): super().__init__(f"Einstein Teleparallel (τ={tau:.1e})"); self.tau=torch.tensor(tau,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        base=1-rs/r
        corr=self.tau*(rs/r)**3
        result = (-base,1/(base-corr+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class EinsteinRegularized(GravitationalTheory):
    def __init__(self): super().__init__("Einstein Regularized Core")
    def get_metric(self, r, M, C, G):
        rs,lp=(2*G*M)/C**2,torch.sqrt(torch.tensor(G*hbar/C**3,device=device,dtype=DTYPE))
        m=1-rs/torch.sqrt(r**2+lp**2)
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Yukawa(GravitationalTheory):
    def __init__(self,lambda_mult): super().__init__(f"Yukawa (λ={lambda_mult:.2f}*RS)"); self.lambda_mult=torch.tensor(lambda_mult,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        m=1-(rs/r)*torch.exp(-r/(self.lambda_mult*rs))
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class QuantumCorrected(GravitationalTheory):
    def __init__(self,alpha): super().__init__(f"Quantum Corrected (α={alpha:.2f})"); self.alpha=torch.tensor(alpha,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        m=1-rs/r+self.alpha*(rs/r)**3
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class HigherDimensional(GravitationalTheory):
    def __init__(self,crossover_mult): super().__init__(f"Higher-Dim (cross={crossover_mult:.2f}*RS)"); self.rc=torch.tensor(crossover_mult*2*G*M.item()/c**2,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs,p4d,p5d=(2*G*M)/C**2,rs/r,(self.rc*rs)/r**2
        t=1/(1+torch.exp(-(r-self.rc)/(self.rc/10)))
        m=1-(t*p4d+(1-t)*p5d)
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class LogCorrected(GravitationalTheory):
    def __init__(self,beta): super().__init__(f"Log Corrected (β={beta:.2f})"); self.beta=torch.tensor(beta,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs,sr=(2*G*M)/C**2,torch.maximum(r,rs*1.001)
        lc=self.beta*(rs/sr)*torch.log(sr/rs)
        m=1-rs/r+lc
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class VariableG(GravitationalTheory):
    def __init__(self,delta): super().__init__(f"Variable G (δ={delta:.2f})"); self.delta=torch.tensor(delta,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs,G_eff=(2*G*M)/C**2,G*(1+self.delta*rs/r)
        m=1-(2*G_eff*M)/(C**2*r)
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class NonLocal(GravitationalTheory):
    def __init__(self): super().__init__("Non-local (Cosmological)")
    def get_metric(self, r, M, C, G):
        rs,ct=(2*G*M)/C**2,torch.tensor(LAMBDA_COSMO,device=device,dtype=DTYPE)*r**2/3
        m=1-rs/r-ct
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Fractal(GravitationalTheory):
    def __init__(self,D): super().__init__(f"Fractal (D={D:.2f})"); self.D=torch.tensor(D,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        m=1-(rs/r)**(self.D-2.0)
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class PhaseTransition(GravitationalTheory):
    def __init__(self,crit_mult): super().__init__(f"Phase Transition(crit={crit_mult:.2f}*RS)"); self.rc=torch.tensor(crit_mult*2*G*M.item()/c**2,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        n=1-rs/r
        co=1-rs/self.rc
        m=torch.where(r>self.rc,n,co)
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Acausal(GravitationalTheory):
    def __init__(self): super().__init__("Acausal (Final State)")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        ht=(hbar*C**3)/(8*torch.pi*G*M*k)
        pt=torch.sqrt(torch.tensor(hbar*C**5/(G*k**2),device=device,dtype=DTYPE))
        cf=1-(ht/pt)
        m=1-(rs*cf)/r
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Computational(GravitationalTheory):
    def __init__(self): super().__init__("Computational Complexity")
    def get_metric(self, r, M, C, G):
        rs,lp=(2*G*M)/C**2,torch.sqrt(torch.tensor(G*hbar/C**3,device=device,dtype=DTYPE))
        sr=torch.maximum(r,lp)
        cf=(rs**2)
        m=1-cf/(sr*torch.log2(sr/lp))
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Tduality(GravitationalTheory):
    def __init__(self): super().__init__("String T-Duality")
    def get_metric(self, r, M, C, G):
        rs,ls=(2*G*M)/C**2,rs
        re=r+ls**2/r
        m=1-rs/re
        result = (-m,1/(m+EPSILON),r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Hydrodynamic(GravitationalTheory):
    def __init__(self): super().__init__("Emergent (Hydrodynamic)")
    def get_metric(self, r, M, C, G):
        rs=(2*G*M)/C**2
        vfs=torch.minimum((rs*C**2)/r,torch.tensor(C**2*0.999999,device=device,dtype=DTYPE))
        gs=1/(1-vfs/C**2+EPSILON)
        result = (-1/gs,gs,r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

class Participatory(GravitationalTheory):
    def __init__(self,obs_energy): super().__init__(f"Participatory(E_obs={obs_energy:.1e})"); self.obs_energy=torch.tensor(obs_energy,device=device,dtype=DTYPE)
    def get_metric(self, r, M, C, G):
        rs,Ep=(2*G*M)/C**2,torch.sqrt(torch.tensor(hbar*C**5/G,device=device,dtype=DTYPE))
        cert=1-torch.exp(-5*self.obs_energy/Ep)
        g_tt_gr,g_rr_gr=-(1-rs/r),1/(1-rs/r+EPSILON)
        g_tt_vac,g_rr_vac=-1.0,1.0
        g_tt=cert*g_tt_gr+(1-cert)*g_tt_vac
        g_rr=cert*g_rr_gr+(1-cert)*g_rr_vac
        result = (g_tt,g_rr,r**2,torch.tensor(0.0,device=device,dtype=DTYPE))
        return result

# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
# E. MAIN SIMULATION SCRIPT
# ===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-===-
if __name__ == "__main__":
    total_start_time = time.time()
    print("Main simulation script started.")

    print("Building list of models to test...")
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
            if DEBUG_COMPUTE:
                print(f"[LOG] Adding model {model_class.__name__} with {param_key}={val}")
            models_to_test.append(model_class(**{param_key: val}))

    total_models = len(models_to_test)
    print(f"Initializing PyTorch GPU orbital test for {total_models} universes...\n")

    r0 = 4.0 * RS
    v_tan = torch.sqrt(G * M / r0)
    g_tt_i,_,g_pp_i,_=Schwarzschild().get_metric(r0, M, c, G)
    dt_dtau_i=torch.sqrt(-c**2/(g_tt_i*c**2+g_pp_i*(v_tan/r0)**2))
    dphi_dtau_i=(v_tan/r0)*dt_dtau_i
    y0 = torch.tensor([0.0,r0.item(),0.0,dt_dtau_i.item(),0.0,dphi_dtau_i.item()],device=device,dtype=DTYPE)

    DT = 1.0; T_MAX = 100_000.0; N_STEPS = int(T_MAX / DT)
    print("Running ground truth simulation with Schwarzschild (GR)...")
    y_true = y0.clone()
    start_time_gt = time.time()
    for step in range(N_STEPS):
        y_true=rk4_step(y_true,geodesic_ode_pytorch,models_to_test[0],DT)
        if (step+1) % 10000 == 0:
            elapsed = time.time() - start_time_gt
            accuracy_loss = torch.abs(y_true[1] - r0) / r0 * 100
            print(f"  Step {(step+1):,}/{N_STEPS:,} ({(step+1)/N_STEPS*100:.1f}%) - r={y_true[1]/RS:.4f}*RS - Accuracy Loss: {accuracy_loss:.6f}% - Elapsed: {elapsed:.1f}s")
        if y_true[1]<=RS: 
            print(f"  Event horizon reached at step {step+1:,}")
            break
    ground_truth_final_state=y_true.cpu().numpy()
    print(f"GR orbit completed. Final r={ground_truth_final_state[1]/RS.cpu():.4f}*RS, φ={np.rad2deg(ground_truth_final_state[2]):.2f}°\n")

    results = []
    print("--- Simulating Orbital Trajectories for All Models (PyTorch GPU) ---")
    for model in models_to_test:
        print(f"Testing: {model.name}...")
        model_start_time = time.time()
        y_pred = y0.clone()
        for i in range(N_STEPS):
            y_pred=rk4_step(y_pred,geodesic_ode_pytorch,model,DT)
            if y_pred[1]<=RS: 
                break

        final_state_pred=y_pred.cpu().numpy()
        loss = np.sqrt((ground_truth_final_state[1]-final_state_pred[1])**2 + (r0.item()*(ground_truth_final_state[2]-final_state_pred[2]))**2)
        if not np.isfinite(loss): 
            loss = 1e50
        results.append({"Model": model.name, "Loss": loss})
        print(f"  -> Finished in {time.time()-model_start_time:.2f}s. Deviation: {loss:.4e} m")

    print("\n--- PYTORCH GPU ORBITAL TEST (FLOAT32): RANKED DEVIATION ---")
    results.sort(key=lambda x: x["Loss"])
    print(f"{'Rank':<5} | {'Model Name':<65} | {'Final Trajectory Deviation (m)':<35}")
    print("="*110)
    for rank, res in enumerate(results, 1):
        loss_str = f"{res['Loss']:.4e}" if res['Loss'] < 1e49 else "Solver Failed or Plunged"
        print(f"{rank:<5} | {res['Model']:<65} | {loss_str}")
    print("="*110)
    print(f"Total execution time: {time.time() - total_start_time:.2f} seconds")