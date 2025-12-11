import matplotlib.pyplot as plt
import numpy as np

def plot_peak_freq(force_history, all_peak_freqs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Force History on ax1
    ax1.plot(force_history)
    ax1.set_title("Force History")
    ax1.set_ylabel("End-effector Force (N)")
    ax1.grid(True)
    ax1.set_ylim(0, 10.0)

    # Plot 2: Peak Frequencies on ax2
    for t, freqs_at_t in enumerate(all_peak_freqs):
        if len(freqs_at_t) > 0:
            ax2.scatter([t] * len(freqs_at_t), freqs_at_t, c="b", marker="o")

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Peak frequency (Hz)")
    ax2.set_title("Peak frequencies over time")

    # Adjust layout and show the combined plot
    plt.tight_layout()
    plt.show()
    
    # # Plot the force history after rightward movement
    # plt.figure(figsize=(10, 4))
    # plt.plot(force_history)
    # plt.title(f"Force History")
    # plt.xlabel("Timestep")
    # plt.ylabel("End-effector Force (N)")
    # plt.grid(True)
    # plt.ylim(0, 10.0)
    # plt.show()

    # for t, freqs_at_t in enumerate(all_peak_freqs):
    #     plt.scatter([t] * len(freqs_at_t), freqs_at_t, c="b", marker="o")

    # plt.xlabel("Time step")
    # plt.ylabel("Peak frequency (Hz)")
    # plt.title("Peak frequencies over time")
    # plt.show()