# fit_lizhi_both_plots.py
import numpy as np
from scipy.optimize import differential_evolution
import control as ctl
import matplotlib.pyplot as plt

# -------------------------
# Constants for the Lizhi Model
Kg = 10.0
K1 = 0.6
K2 = 0.4

# -------------------------
# Parameters for the Target TGOV1 Model
R = 0.33
T1_r = 2.0
T2_r = 3.0
T3_r = 15.0
Dt = 0.4

# -------------------------
# Frequency grid for comparison (rad/s)
w_full = np.logspace(-4, 4, 500)

# -------------------------
# User-defined target frequency domain (for optimization + zoomed plot)
w_target_min = 1e-1   # <-- user can change
w_target_max = 1e2    # <-- user can change

# -------------------------
# Helper function to get frequency response
def freq_resp_db_deg(sys, w):
    mag, phase, _ = ctl.bode(sys, w, plot=False)
    mag_db = 20 * np.log10(mag + 1e-30)
    phase_deg = np.rad2deg(phase)
    return mag_db, phase_deg

# -------------------------
# TGOV1 transfer function
def create_tgov1_dw_tf(R, T1, T2, T3, Dt):
    num1 = [T2, 1.0]
    num2_poly = np.polymul([T1, 1.0], [T3, 1.0])
    num2 = Dt * R * num2_poly
    num = -np.polyadd(num1, num2)
    den = R * num2_poly
    return ctl.tf(num, den)

# -------------------------
# Lizhi model transfer function
def Lizhi_dw_tf(params):
    T1, T2, T3, T4, T5 = params
    num_A = [Kg * T2, Kg]
    den_A_1 = np.polymul([T1, 1.0], [T3, 1.0])
    den_A = np.polymul(den_A_1, [T4, 1.0])
    G_A = ctl.tf(num_A, den_A)
    num_B = [K1 * T5, K1 + K2]
    den_B = [T5, 1.0]
    G_B = ctl.tf(num_B, den_B)
    G_lizhi_ref = G_A * G_B
    G_dt = ctl.tf([Dt], [1.0])
    return -(G_lizhi_ref + G_dt)

# -------------------------
# Cost function with frequency-domain weighting
def cost(params, w_target_min=None, w_target_max=None, weight_inside=2.0, weight_outside=1.0):
    if np.any(np.array(params) <= 0):
        return 1e9
    
    G_target = create_tgov1_dw_tf(R, T1_r, T2_r, T3_r, Dt)
    G_model = Lizhi_dw_tf(params)
    mag_target, ph_target = freq_resp_db_deg(G_target, w_full)
    mag_model, ph_model = freq_resp_db_deg(G_model, w_full)

    # Compute errors
    mag_err_arr = (mag_model - mag_target) ** 2
    ph_err_arr = (ph_model - ph_target) ** 2

    # Initialize weights (default outside)
    weights = np.ones_like(w_full) * weight_outside

    # Apply higher weight inside the target region
    if w_target_min is not None and w_target_max is not None:
        mask = (w_full >= w_target_min) & (w_full <= w_target_max)
        weights[mask] = weight_inside

    # Weighted mean error
    mag_err = np.mean(weights * mag_err_arr)
    ph_err = np.mean(weights * ph_err_arr)

    return 1.0 * mag_err + 1.0 * ph_err

# -------------------------
# Optimization bounds and execution
bounds = [(1e-3, 50.0)] * 5
result = differential_evolution(
    lambda p: cost(p, w_target_min=w_target_min, w_target_max=w_target_max, weight_inside=3.0, weight_outside=1.0),
    bounds,
    maxiter=200, popsize=15, tol=1e-6, polish=True,
    disp=True, updating='deferred', seed=32
)

T_optimal = result.x
print("Optimized T1..T5:", T_optimal)

# -------------------------
# Compute frequency responses
G_target_final = create_tgov1_dw_tf(R, T1_r, T2_r, T3_r, Dt)
G_lizhi_fitted = Lizhi_dw_tf(T_optimal)
mag_target, ph_target = freq_resp_db_deg(G_target_final, w_full)
mag_fitted, ph_fitted = freq_resp_db_deg(G_lizhi_fitted, w_full)

# -------------------------
# Select zoomed region = user-specified target domain
mask_zoom = (w_full >= w_target_min) & (w_full <= w_target_max)
w_zoom = w_full[mask_zoom]
mag_target_zoom = mag_target[mask_zoom]
mag_fitted_zoom = mag_fitted[mask_zoom]
ph_target_zoom = ph_target[mask_zoom]
ph_fitted_zoom = ph_fitted[mask_zoom]

# -------------------------
# Plot 1: Full frequency Bode plot
plt.style.use('seaborn-v0_8-whitegrid')
fig1, (ax1f, ax2f) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig1.suptitle('Bode Plot: Full Frequency Range', fontsize=16)

ax1f.semilogx(w_full, mag_target, 'b-', label='Target (TGOV1)', linewidth=2)
ax1f.semilogx(w_full, mag_fitted, 'r--', label='Fitted (Lizhi)', linewidth=2)
ax1f.set_ylabel('Magnitude (dB)')
ax1f.grid(True, which='both')
ax1f.legend()

ax2f.semilogx(w_full, ph_target, 'b-', label='Target (TGOV1)', linewidth=2)
ax2f.semilogx(w_full, ph_fitted, 'r--', label='Fitted (Lizhi)', linewidth=2)
ax2f.set_ylabel('Phase (deg)')
ax2f.set_xlabel('Frequency (rad/s)')
ax2f.grid(True, which='both')
ax2f.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# -------------------------
# Plot 2: Zoomed target region
fig2, (ax1z, ax2z) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig2.suptitle(f'Bode Plot: Target Region {w_target_min:.2e} – {w_target_max:.2e} rad/s', fontsize=16)

ax1z.semilogx(w_zoom, mag_target_zoom, 'b-', label='Target (TGOV1)', linewidth=2)
ax1z.semilogx(w_zoom, mag_fitted_zoom, 'r--', label='Fitted (Lizhi)', linewidth=2)
ax1z.set_ylabel('Magnitude (dB)')
ax1z.grid(True, which='both')
ax1z.legend()

ax2z.semilogx(w_zoom, ph_target_zoom, 'b-', label='Target (TGOV1)', linewidth=2)
ax2z.semilogx(w_zoom, ph_fitted_zoom, 'r--', label='Fitted (Lizhi)', linewidth=2)
ax2z.set_ylabel('Phase (deg)')
ax2z.set_xlabel('Frequency (rad/s)')
ax2z.grid(True, which='both')
ax2z.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
