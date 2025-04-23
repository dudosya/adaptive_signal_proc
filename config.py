# config.py
"""
Configuration file for the Regularized Adaptive Long Autoregressive Spectral Analysis simulation.
"""

import numpy as np

# --- Simulation Parameters ---
M = 110  # Number of range bins
N = 8    # Number of samples per bin (pulses)

# --- True Signal Generation Parameters ---
# Define characteristics for different signal types
# Format: (start_bin, end_bin, ar_coeffs, noise_variance)
# Bins are inclusive, 0-indexed. end_bin is the index of the last bin in the range.
# Note: Order matters, later definitions overwrite earlier ones in overlapping regions.
SIGNAL_DEFINITIONS = [
    # Default: AR(0) White Noise (covers all bins initially)
    (0, M-1, np.array([], dtype=complex), 0.5),

    # Ground Clutter: AR(1), pole near 1.0
    (15, 56, np.array([-0.97 + 0j]), 0.7), # Bins 15 to 56 inclusive

    # Rain Clutter: AR(1), complex pole
    (35, 74, np.array([-0.88 * np.exp(1j * np.pi * 0.1)]), 1.0), # Bins 35 to 74 inclusive

    # Sea Echoes: AR(2), complex conjugate poles
    (56, 94, np.array([-(0.95*np.exp(1j*np.pi*0.3) + 0.95*np.exp(-1j*np.pi*0.3)),
                       (0.95*np.exp(1j*np.pi*0.3)) * (0.95*np.exp(-1j*np.pi*0.3))]), 1.3) # Bins 56 to 94 inclusive
]

# --- Spectral Analysis Parameters ---
FS = 1.0       # Sampling frequency (use 1.0 for normalized frequency plots)
NFFT = 1024    # FFT points for spectral estimation plots

# --- Standard LS Method Parameters ---
P_LS = 3       # AR order for the standard Least Squares method

# --- Regularized LS (Kalman Smoother) Method Parameters ---
P_REG = N - 1  # AR order for the Regularized method (often N-1)

# Smoothness order k for regularization penalties
K_SMOOTHNESS = 1

# Regularization hyperparameters (PLACEHOLDERS - Set manually or use ML estimation)
# High lambda_d => Strong spatial smoothing
# High lambda_s => Strong spectral smoothing
LAMBDA_S = 0.1   # Spectral smoothness weight (1/r_s)
LAMBDA_D = 10.0  # Spatial continuity weight (1/r_d)

# --- Kalman Filter/Smoother Parameters ---
# Initial state covariance magnitude (P0 = P0_MAG * I)
P0_MAG = 1e6

# Regularization added to YHY matrix before inversion (for numerical stability)
YHY_REG = 1e-8

# --- Plotting Parameters ---
# Set common color limits (vmin, vmax) for PSD plots in dB.
# Set to None for automatic scaling.
PLOT_VMIN_DB = -60
PLOT_VMAX_DB = None # Example: Use auto-scaling for max, or set a value e.g., 30

# Figure size for the final 4-panel comparison plot
FINAL_PLOT_FIGSIZE = (26, 6)

# --- Output Parameters ---
# Set to True to save intermediate results or final plots
SAVE_PLOTS = False
PLOT_OUTPUT_DIR = "plots" # Directory to save plots if SAVE_PLOTS is True

# --- Miscellaneous ---
# Small epsilon value to prevent log(0) or division by zero
EPSILON = 1e-15