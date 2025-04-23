# main.py

"""
Main script to run the Regularized Adaptive Long Autoregressive Spectral Analysis simulation.
Steps:
1. Load configuration.
2. Generate simulated data.
3. Perform standard LS estimation.
4. Perform Regularized LS estimation using Kalman Smoother.
5. Calculate spectra for all methods.
6. Plot results for comparison.
"""

import numpy as np
import time
import os
from scipy.signal import periodogram

# Import configuration parameters
import config

# Import utility functions
import utils

# Import plotting functions
import plot

def main():
    """ Main execution function """
    print("--- Starting Simulation ---")
    start_time = time.time()

    # --- 1. Generate Data ---
    Y_data, true_coeffs, true_noise_vars = utils.generate_all_bins_data(
        config.M, config.N, config.SIGNAL_DEFINITIONS
    )

    # Optional: Plot input data
    if config.SAVE_PLOTS:
        input_plot_filename = os.path.join(config.PLOT_OUTPUT_DIR, "01_input_data.png")
        plot.plot_input_data(Y_data, config.N, config.M, filename=input_plot_filename)
    # else:
    #     plot.plot_input_data(Y_data, config.N, config.M) # Show plot interactively

    # --- 2. Calculate True Spectra & Periodogram ---
    print("\nCalculating baseline spectra...")
    # True Spectra
    true_spectra_db, freq_axis = utils.calculate_all_spectra(
        config.M, config.NFFT, config.FS,
        true_coeffs, # Note: Need to handle list of arrays correctly
        true_noise_vars
    )
    # Need to convert true_coeffs list to an array for calculate_all_spectra if needed,
    # or modify calculate_all_spectra to handle list (let's modify utils slightly)

    # Recalculate true spectra with updated utils function (if modified)
    # Re-fetch true_coeffs/vars if generate_all_bins_data was changed
    # Assuming utils.calculate_all_spectra handles the list format now
    true_coeffs_array = np.array(true_coeffs, dtype=object) # Not ideal, better handled inside util
    # Let's assume calculate_all_spectra is modified to handle list of coeff arrays

    # Periodogram - calculate directly here or add a util function
    periodogram_db = np.zeros((config.M, config.NFFT))
    print("Calculating periodograms...")
    for m in range(config.M):
        f, Pxx = periodogram(Y_data[m, :], fs=config.FS, nfft=config.NFFT, return_onesided=False)
        Pxx_shifted = np.fft.fftshift(Pxx)
        periodogram_db[m, :] = 10 * np.log10(Pxx_shifted + config.EPSILON)
        if m == 0: # Get freq axis once
             freq_axis_p = np.fft.fftshift(f) # Already shifted in calculate_ar_spectrum_db
             freq_axis_norm = np.fft.fftshift(np.fft.fftfreq(config.NFFT, 1/config.FS))

    # Use the frequency axis calculated by calculate_ar_spectrum_db
    # freq_axis = freq_axis_norm # Or use the one from the util function output

    # --- 3. Standard LS Estimation ---
    ls_coeffs, ls_noise_vars = utils.run_standard_ls_estimation(Y_data, config.P_LS)
    ls_spectra_db, _ = utils.calculate_all_spectra(
        config.M, config.NFFT, config.FS,
        ls_coeffs,
        ls_noise_vars
    )

    # --- 4. Regularized LS Estimation (Kalman Smoother) ---
    print("\nStarting Regularized LS (Kalman Smoother)...")
    ks_start_time = time.time()

    # 4a. Calculate KS parameters
    F_k, Q_k, Delta_k = utils.calculate_ks_parameters(
        config.LAMBDA_S, config.LAMBDA_D, config.P_REG, config.K_SMOOTHNESS
    )

    # 4b. Calculate effective measurements
    # Use noise vars from low-order LS as initial weights/variances r_m^e
    LS_estimates_preg, R_eff = utils.calculate_effective_measurements(
        Y_data, config.P_REG, ls_noise_vars
    )

    # 4c. Run Kalman Filter (Forward Pass)
    a0 = np.zeros(config.P_REG, dtype=complex)
    P0 = np.identity(config.P_REG) * config.P0_MAG
    A_filtered, P_filtered, A_predicted, P_predicted = utils.run_kalman_filter(
        config.M, config.P_REG, F_k, Q_k, LS_estimates_preg, R_eff, a0, P0
    )

    # 4d. Run RTS Smoother (Backward Pass)
    A_smoothed, P_smoothed = utils.run_rts_smoother(
        config.M, config.P_REG, F_k,
        A_filtered, P_filtered, A_predicted, P_predicted
    )

    ks_end_time = time.time()
    print(f"Regularized LS (Kalman Smoother) took {ks_end_time - ks_start_time:.2f} seconds.")

    # --- 5. Calculate Final RegLS Spectra ---
    # Recalculate noise variance using smoothed coefficients
    regls_noise_vars = np.zeros(config.M)
    for m in range(config.M):
        residual_sq = utils.calculate_residual_norm_sq(Y_data[m, :], A_smoothed[m, :], config.P_REG)
        regls_noise_vars[m] = np.maximum(config.EPSILON, np.real(residual_sq / config.N))

    # Calculate spectra
    regls_spectra_db, _ = utils.calculate_all_spectra(
        config.M, config.NFFT, config.FS,
        A_smoothed,
        regls_noise_vars
    )

    # --- 6. Plot Final Comparison ---
    final_plot_filename = None
    if config.SAVE_PLOTS:
        final_plot_filename = os.path.join(config.PLOT_OUTPUT_DIR, "02_spectrum_comparison.png")

    # Need to ensure true_spectra_db was calculated correctly
    # Re-do true spectra calc cleanly:
    true_coeffs_for_spec = [np.array(c) for c in true_coeffs] # Ensure they are arrays
    true_spectra_db, freq_axis_final = utils.calculate_all_spectra(
         config.M, config.NFFT, config.FS,
         true_coeffs_for_spec, # Pass list of arrays
         true_noise_vars
    )
    # Make sure the frequency axis is consistent
    if freq_axis_final is None:
         print("ERROR: Failed to get frequency axis.")
         return

    plot.plot_spectrum_comparison(
        freq_axis_final, config.M, config.N, config.P_LS, config.P_REG,
        true_spectra_db,
        periodogram_db,
        ls_spectra_db,
        regls_spectra_db,
        vmin=config.PLOT_VMIN_DB, vmax=config.PLOT_VMAX_DB,
        filename=final_plot_filename
    )

    # --- End ---
    end_time = time.time()
    print("\n--- Simulation Complete ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

# --- Entry Point ---
if __name__ == "__main__":
    # Add argument parsing here later if needed (e.g., override config values)
    main()