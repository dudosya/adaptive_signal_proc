# utils.py

"""
Utility functions for the Regularized Adaptive Long Autoregressive Spectral Analysis simulation.
Contains functions for:
- Data generation
- Autocorrelation estimation
- LS AR estimation (Yule-Walker)
- Kalman Filter/Smoother steps
- Residual calculation
- Spectrum calculation from AR parameters
"""

import numpy as np
from scipy.signal import lfilter, freqz
from scipy.linalg import toeplitz, solve, inv, LinAlgError

# Import constants from config - assumes config.py is in the same directory
# or the Python path is set up correctly.
import config

# --- True Data Generation ---

def generate_single_bin_ar_data(ar_coeffs, noise_variance, n_samples):
    """
    Generates N complex data samples for a single bin from an AR process.
    Uses the convention: y[n] = -sum(a[p-1]*y[n-p]) + e[n].
    """
    p = len(ar_coeffs)
    noise_std_dev = np.sqrt(noise_variance / 2.0)
    real_noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=n_samples)
    imag_noise = np.random.normal(loc=0.0, scale=noise_std_dev, size=n_samples)
    complex_noise = real_noise + 1j * imag_noise
    ar_filter_coeffs = np.concatenate(([1], ar_coeffs))
    y = lfilter(b=[1], a=ar_filter_coeffs, x=complex_noise)
    return y

def generate_all_bins_data(M, N, signal_definitions):
    """
    Generates the MxN data matrix based on signal definitions.
    """
    print("Generating true AR parameters and data...")
    true_ar_coeffs_list = [np.array([], dtype=complex)] * M
    true_noise_var_array = np.zeros(M)

    # Apply signal definitions - order matters for overlaps
    for start_bin, end_bin, coeffs, variance in signal_definitions:
        for m in range(start_bin, end_bin + 1):
            if 0 <= m < M:
                true_ar_coeffs_list[m] = coeffs
                true_noise_var_array[m] = variance

    Y_data = np.zeros((M, N), dtype=complex)
    for m in range(M):
        coeffs = true_ar_coeffs_list[m]
        variance = true_noise_var_array[m]
        Y_data[m, :] = generate_single_bin_ar_data(coeffs, variance, N)

    print("Data matrix Y_data generated.")
    return Y_data, true_ar_coeffs_list, true_noise_var_array

# --- Autocorrelation and Residuals ---

def estimate_autocorrelation(y_data, max_lag):
    """
    Estimates biased autocorrelation R(k) = E[y(n) y(n-k)*] / N
    for lags k = 0 to max_lag.
    """
    N_local = len(y_data)
    y_padded = np.concatenate((y_data, np.zeros(max_lag)))
    R = np.zeros(max_lag + 1, dtype=complex)
    for k in range(max_lag + 1):
        R[k] = np.dot(y_padded[k:k+N_local], np.conj(y_padded[:N_local])) / N_local
    return R

def calculate_residual_norm_sq(y_data, a_m, order):
    """ Calculates ||y_m - Y_m a_m||^2 using autocorrelation estimates """
    N_local = len(y_data)
    if len(a_m) != order:
         # Handle case where order is 0 (a_m is empty)
         if order == 0 and len(a_m) == 0:
             R = estimate_autocorrelation(y_data, 0)
             return np.real(N_local * R[0]) # ||y||^2
         else:
            raise ValueError(f"Length of a_m ({len(a_m)}) must match order ({order})")

    R = estimate_autocorrelation(y_data, order)
    R0 = np.real(R[0])
    r_vec = R[1:order+1]
    R_mat = toeplitz(R[:order])

    term1 = N_local * R0
    term2 = -N_local * np.dot(np.conj(r_vec), a_m)
    term3 = -N_local * np.dot(np.conj(a_m), r_vec)
    term4 = N_local * np.dot(np.conj(a_m), np.dot(R_mat, a_m))

    residual_norm_sq = term1 + term2 + term3 + term4
    return np.real(residual_norm_sq)

# --- Standard LS Estimation ---

def ls_ar_yule_walker(y_data, order):
    """ Estimates AR parameters using the Yule-Walker method for a single bin. """
    N_local = len(y_data)
    if N_local <= order: # Use <= to be safe
        print(f"Warning: Data length {N_local} <= order {order} for bin. Cannot estimate reliably.")
        return np.zeros(order, dtype=complex), config.EPSILON, np.nan # Return small noise_var

    R = estimate_autocorrelation(y_data, order)
    R0 = np.real(R[0])
    if R0 < config.EPSILON: # Handle zero-energy signal
         return np.zeros(order, dtype=complex), config.EPSILON, R0

    R_matrix = toeplitz(R[:order])
    r_vector = R[1:order+1]

    try:
        a_m = solve(R_matrix, -r_vector, assume_a='her')
    except LinAlgError:
        # print("Warning: Yule-Walker matrix singular. Using pseudo-inverse.")
        try:
            R_inv = np.linalg.pinv(R_matrix)
            a_m = -R_inv @ r_vector
        except LinAlgError:
             print("ERROR: Pseudo-inverse also failed for YW matrix. Returning zeros.")
             a_m = np.zeros(order, dtype=complex)


    noise_var_est = R[0] + np.dot(a_m, np.conj(r_vector))
    noise_var_est = np.maximum(config.EPSILON, np.real(noise_var_est)) # Ensure positive

    return a_m, noise_var_est, R0

def run_standard_ls_estimation(Y_data, P_ls):
    """ Runs standard LS estimation for all bins. """
    print(f"Running Standard LS Estimation (P={P_ls})...")
    M = Y_data.shape[0]
    all_ls_coeffs = np.zeros((M, P_ls), dtype=complex)
    all_ls_noise_vars = np.zeros(M)

    for m in range(M):
        coeffs, noise_var, _ = ls_ar_yule_walker(Y_data[m, :], P_ls)
        all_ls_coeffs[m, :] = coeffs
        all_ls_noise_vars[m] = noise_var

    print("Standard LS Estimation complete.")
    return all_ls_coeffs, all_ls_noise_vars

# --- Regularized LS Estimation (Kalman Smoother Steps) ---

def calculate_ks_parameters(lambda_s, lambda_d, P_reg, k_smoothness):
    """ Calculates stationary KS parameters F, Q, alpha_inf """
    print("Calculating stationary KS parameters...")
    rho = lambda_s / (lambda_d + config.EPSILON)
    theta = 2.0 + rho
    sqrt_term = np.sqrt(np.maximum(0, theta**2 - 4.0))
    alpha_inf = (theta - sqrt_term) / 2.0
    alpha_inf = np.clip(alpha_inf, 0, 1) # Ensure valid range

    F_k = alpha_inf * np.identity(P_reg)

    r_d = 1.0 / (lambda_d + config.EPSILON)
    r_inf = r_d * alpha_inf

    p_indices = np.arange(1, P_reg + 1)
    delta_k_diag = p_indices**(2 * k_smoothness)
    Delta_k = np.diag(delta_k_diag)

    try:
        Delta_k_inv = inv(Delta_k)
    except LinAlgError:
        print("Warning: Delta_k matrix singular. Using pseudo-inverse.")
        Delta_k_inv = np.linalg.pinv(Delta_k)

    Q_k = r_inf * Delta_k_inv
    print(f"alpha_inf={alpha_inf:.4f}, F={alpha_inf:.4f}*I, Q calculated.")
    return F_k, Q_k, Delta_k # Return Delta_k as it might be needed

def calculate_effective_measurements(Y_data, P_reg, initial_noise_vars):
    """ Calculates LS estimates (P=P_reg) and R_eff for all bins """
    print(f"Calculating effective measurements (LS P={P_reg} and R_eff)...")
    M, N = Y_data.shape
    all_R_eff = np.zeros((M, P_reg, P_reg), dtype=complex)
    all_LS_estimates_preg = np.zeros((M, P_reg), dtype=complex)
    failed_bins = []

    for m in range(M):
        y_m = Y_data[m, :]
        r_me_est = initial_noise_vars[m] # Use initial estimate for weighting

        R_m_full = estimate_autocorrelation(y_m, P_reg)
        R_matrix_m = toeplitz(R_m_full[:P_reg])
        YHY_inv_m = np.zeros((P_reg, P_reg), dtype=complex)
        a_ls_m = np.zeros(P_reg, dtype=complex)

        try:
            YHY_m = N * R_matrix_m
            YHY_inv_m = inv(YHY_m + config.YHY_REG * np.identity(P_reg))
        except LinAlgError:
            try:
                 YHY_inv_m = np.linalg.pinv(YHY_m + config.YHY_REG * np.identity(P_reg)) # Try pinv with reg
            except LinAlgError:
                 print(f"ERROR: (P)Inv failed for YHY_m bin {m}. R_eff/LS kept as zeros.")
                 failed_bins.append(m)

        if m not in failed_bins:
            r_me_safe = np.maximum(config.EPSILON, r_me_est)
            all_R_eff[m, :, :] = r_me_safe * YHY_inv_m

            r_vector_m = R_m_full[1:P_reg+1]
            try:
                a_ls_m = -N * (YHY_inv_m @ r_vector_m)
            except Exception as e:
                print(f"Warning: LS (P={P_reg}) vector calculation failed for bin {m}: {e}")
                failed_bins.append(m)

        all_LS_estimates_preg[m, :] = a_ls_m

    if failed_bins:
         print(f"Warning: R_eff/LS calculation had issues for {len(set(failed_bins))} bins.")
    print("Effective measurements calculation complete.")
    return all_LS_estimates_preg, all_R_eff

def run_kalman_filter(M, P_reg, F_k, Q_k, z_meas, R_meas, a0, P0):
    """ Runs the Kalman Filter forward pass, returns filtered and predicted states/covs """
    print("Running Kalman Filter forward pass...")
    A_filtered = np.zeros((M, P_reg), dtype=complex)
    P_filtered = np.zeros((M, P_reg, P_reg), dtype=complex)
    A_predicted = np.zeros((M, P_reg), dtype=complex)
    P_predicted = np.zeros((M, P_reg, P_reg), dtype=complex)

    a_prev = a0
    P_prev = P0
    H_k = np.identity(P_reg)
    I_P = np.identity(P_reg)

    for m in range(M):
        a_pred = F_k @ a_prev
        P_pred = F_k @ P_prev @ F_k.T + Q_k
        A_predicted[m, :] = a_pred
        P_predicted[m, :, :] = P_pred

        z_m = z_meas[m, :]
        R_m = R_meas[m, :, :]

        if np.linalg.norm(R_m) < config.EPSILON**2: # Check norm squared vs epsilon^2
             # print(f"Warning: Skipping update for bin {m} due to potentially invalid R_m.")
             a_filt = a_pred
             P_filt = P_pred
        else:
            S_m = P_pred + R_m
            try:
                 # K = P H^T S^-1 = P S^-1 (H=I) => S^H K^T = P^H => S K^T = P (S, P herm)
                 K_m = solve(S_m, P_pred, assume_a='her').T
            except LinAlgError:
                 # print(f"Warning: S_m singular bin {m}. Using pinv.")
                 try:
                     S_m_inv = np.linalg.pinv(S_m)
                     K_m = P_pred @ S_m_inv
                 except LinAlgError:
                      # print(f"ERROR: pinv(S_m) failed bin {m}. Skipping update.")
                      K_m = np.zeros_like(P_pred)

            v_m = z_m - a_pred # H=I
            a_filt = a_pred + K_m @ v_m
            ImKH = I_P - K_m # H=I
            P_filt = ImKH @ P_pred @ ImKH.conj().T + K_m @ R_m @ K_m.conj().T
            P_filt = 0.5 * (P_filt + P_filt.conj().T) # Enforce symmetry

        A_filtered[m, :] = a_filt
        P_filtered[m, :, :] = P_filt
        a_prev = a_filt
        P_prev = P_filt

    print("Kalman Filter forward pass complete.")
    return A_filtered, P_filtered, A_predicted, P_predicted


def run_rts_smoother(M, P_reg, F_k, A_filtered, P_filtered, A_predicted, P_predicted):
    """ Runs the RTS Smoother backward pass """
    print("Running RTS Smoother backward pass...")
    A_smoothed = np.zeros_like(A_filtered)
    P_smoothed = np.zeros_like(P_filtered)

    A_smoothed[M-1, :] = A_filtered[M-1, :]
    P_smoothed[M-1, :, :] = P_filtered[M-1, :, :]

    for m in range(M-2, -1, -1):
        a_filt_m = A_filtered[m, :]
        P_filt_m = P_filtered[m, :, :]
        a_pred_m_plus_1 = A_predicted[m+1, :]
        P_pred_m_plus_1 = P_predicted[m+1, :, :]
        a_smooth_m_plus_1 = A_smoothed[m+1, :]
        P_smooth_m_plus_1 = P_smoothed[m+1, :, :]

        # G = P_filt F^H P_pred_inv => P_pred^H G = F^H P_filt^H => P_pred G = F P_filt (Hermitian, F real)
        try:
            G_m = solve(P_pred_m_plus_1, (F_k @ P_filt_m), assume_a='her')
        except LinAlgError:
            # print(f"Warning: P_pred[{m+1}|{m}] singular smoother bin {m}. Using pinv.")
            try:
                P_pred_inv = np.linalg.pinv(P_pred_m_plus_1)
                G_m = F_k @ P_filt_m @ P_pred_inv # G = F P P_pred_inv (F=F^H)
            except LinAlgError:
                # print(f"ERROR: pinv(P_pred) failed smoother bin {m}. Setting Gain=0.")
                G_m = np.zeros((P_reg, P_reg))

        a_smooth_m = a_filt_m + G_m @ (a_smooth_m_plus_1 - a_pred_m_plus_1)
        P_smooth_m = P_filt_m + G_m @ (P_smooth_m_plus_1 - P_pred_m_plus_1) @ G_m.conj().T
        P_smooth_m = 0.5 * (P_smooth_m + P_smooth_m.conj().T)

        A_smoothed[m, :] = a_smooth_m
        P_smoothed[m, :, :] = P_smooth_m

    print("RTS Smoother backward pass complete.")
    return A_smoothed, P_smoothed

# --- Spectrum Calculation ---

def calculate_ar_spectrum_db(ar_coeffs, noise_variance, nfft, fs):
    """ Calculates the PSD in dB for given AR parameters """
    P_model = len(ar_coeffs)
    if np.isnan(noise_variance) or noise_variance <= 0:
        # print(f"Warning: Invalid noise variance ({noise_variance}). Setting spectrum to zero.")
        psd_db_shifted = np.full(nfft, -150.0) # Assign very low dB value
    else:
        ar_filter_coeffs = np.concatenate(([1], ar_coeffs))
        numerator_b = np.array([np.sqrt(noise_variance)])
        w, h = freqz(b=numerator_b, a=ar_filter_coeffs, worN=nfft, whole=True, fs=fs)
        psd = np.abs(h)**2
        psd_shifted = np.fft.fftshift(psd)

        if psd_shifted.shape[0] != nfft:
             print(f"Warning: Unexpected PSD shape {psd_shifted.shape}. Expected {nfft}.")
             psd_db_shifted = np.full(nfft, -150.0)
        else:
             psd_db_shifted = 10 * np.log10(psd_shifted + config.EPSILON)

    freq_plot_shifted = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs))
    return psd_db_shifted, freq_plot_shifted


def calculate_all_spectra(M, nfft, fs, coeffs_list, noise_vars_list):
    """ Calculates spectra in dB for all M bins given coefficients and variances """
    # Inside utils.calculate_all_spectra
    # ...
    # Determine P - needs care if list
    if isinstance(coeffs_list, list):
        P = len(coeffs_list[0]) if M > 0 and len(coeffs_list[0]) > 0 else 0
    elif coeffs_list.ndim > 1:
        P = coeffs_list.shape[1]
    else: # Should not happen if M > 0
        P = 0
    print(f"Calculating {M} spectra (P={P})...")
    # ... loop using coeffs_list[m] ...
    all_spectra_db = np.zeros((M, nfft))
    freq_axis = None

    for m in range(M):
        spec_db, freqs = calculate_ar_spectrum_db(coeffs_list[m], noise_vars_list[m], nfft, fs)
        all_spectra_db[m, :] = spec_db
        if freq_axis is None:
            freq_axis = freqs

    print("Spectrum calculation complete.")
    return all_spectra_db, freq_axis