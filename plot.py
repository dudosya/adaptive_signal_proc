# plot.py

"""
Plotting functions for the Regularized Adaptive Long Autoregressive Spectral Analysis simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import config settings
import config

def plot_input_data(Y_data, N, M, filename=None):
    """ Plots the real and imaginary parts of the input data Y_data """
    print("Plotting input data...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im_real = axs[0].imshow(np.real(Y_data), aspect='auto', origin='lower',
                            extent=[0, N-1, 0, M-1], cmap='viridis')
    axs[0].set_title("Simulated Data (Real Part)")
    axs[0].set_xlabel("Sample Index (n)")
    axs[0].set_ylabel("Range Bin (m)")
    fig.colorbar(im_real, ax=axs[0], label='Amplitude')

    im_imag = axs[1].imshow(np.imag(Y_data), aspect='auto', origin='lower',
                            extent=[0, N-1, 0, M-1], cmap='viridis')
    axs[1].set_title("Simulated Data (Imaginary Part)")
    axs[1].set_xlabel("Sample Index (n)")
    axs[1].set_ylabel("Range Bin (m)")
    fig.colorbar(im_imag, ax=axs[1], label='Amplitude')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"Input data plot saved to {filename}")
        plt.close(fig) # Close figure after saving if filename provided
    else:
        plt.show()


def plot_spectrum_comparison(freq_axis, M, N, P_ls, P_reg,
                             true_spectra_db,
                             periodogram_db,
                             ls_spectra_db,
                             regls_spectra_db,
                             vmin=None, vmax=None,
                             filename=None):
    """ Plots the 4-panel comparison of spectral estimation methods """
    print("Plotting final spectrum comparison...")
    fig, axs = plt.subplots(1, 4, figsize=config.FINAL_PLOT_FIGSIZE, sharey=True)

    # Extent for imshow
    plot_extent = [freq_axis.min(), freq_axis.max(), 0, M-1]

    # Panel 1: True Spectra
    im0 = axs[0].imshow(true_spectra_db, aspect='auto', origin='lower',
                        extent=plot_extent, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title("True PSD (Ground Truth)")
    axs[0].set_xlabel("Normalized Frequency")
    axs[0].set_ylabel("Range Bin (m)")
    fig.colorbar(im0, ax=axs[0], label='PSD (dB)')

    # Panel 2: Periodograms
    im1 = axs[1].imshow(periodogram_db, aspect='auto', origin='lower',
                        extent=plot_extent, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Periodograms (N={N})")
    axs[1].set_xlabel("Normalized Frequency")
    fig.colorbar(im1, ax=axs[1], label='PSD (dB)')

    # Panel 3: LS Estimated Spectra
    im2 = axs[2].imshow(ls_spectra_db, aspect='auto', origin='lower',
                       extent=plot_extent, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].set_title(f"LS Estimate (P={P_ls}, N={N})")
    axs[2].set_xlabel("Normalized Frequency")
    fig.colorbar(im2, ax=axs[2], label='PSD (dB)')

    # Panel 4: RegLS Estimated Spectra
    im3 = axs[3].imshow(regls_spectra_db, aspect='auto', origin='lower',
                        extent=plot_extent, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[3].set_title(f"RegLS Estimate (P={P_reg}, N={N})")
    axs[3].set_xlabel("Normalized Frequency")
    fig.colorbar(im3, ax=axs[3], label='PSD (dB)')

    plt.suptitle("Comparison of Spectral Estimation Methods", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if filename:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        plt.savefig(filename)
        print(f"Comparison plot saved to {filename}")
        plt.close(fig) # Close figure after saving if filename provided
    else:
        plt.show()

def plot_single_spectrum(freq_axis, psd_db, title, filename=None):
    """ Plots a single spectrum (e.g., periodogram or true spectrum) """
    print(f"Plotting: {title}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    M = psd_db.shape[0]
    plot_extent = [freq_axis.min(), freq_axis.max(), 0, M-1]

    im = ax.imshow(psd_db, aspect='auto', origin='lower',
                    extent=plot_extent, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Range Bin (m)")
    fig.colorbar(im, ax=ax, label='PSD (dB)')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"Plot '{title}' saved to {filename}")
        plt.close(fig)
    else:
        plt.show()