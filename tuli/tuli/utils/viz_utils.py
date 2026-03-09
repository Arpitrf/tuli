import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

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


def plot_rollout_video(
    rgb_frames,              # list of HWC RGB arrays
    force_history,
    all_peak_freqs,
    contact_history,
    output_video_path,
    fps=15,
):

    T = len(force_history)
    assert T == len(rgb_frames), "RGB frames and plot data must have same length"

    # ---------------------------
    # Figure: 1 row, 2 columns
    # Left  = video
    # Right = your 3 vertical subplots
    # ---------------------------
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # ---------------------------
    # LEFT: RGB VIDEO PANEL
    # ---------------------------
    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.axis("off")
    video_im = ax_video.imshow(rgb_frames[0])  # will be updated each frame

    # ---------------------------
    # RIGHT: THREE STACKED SUBPLOTS
    # ---------------------------
    right = gs[0, 1].subgridspec(3, 1, hspace=0.45)
    ax1 = fig.add_subplot(right[0, 0])
    ax2 = fig.add_subplot(right[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(right[2, 0], sharex=ax1)

    # Fix layout manually so they don't overlap
    fig.subplots_adjust(hspace=0.45, left=0.05, right=0.97)

    # ---------------------------
    # AX1 — Force History
    # ---------------------------
    line_force, = ax1.plot([], [], lw=2)
    ax1.set_title("Force History")
    ax1.set_ylabel("End-effector Force (N)")
    ax1.set_ylim(0, 100.0)
    ax1.set_xlim(0, T)
    ax1.grid(True)

    # ---------------------------
    # AX2 — Peak Frequencies
    # ---------------------------
    scatter = ax2.scatter([], [])
    ax2.set_title("Peak frequencies over time")
    ax2.set_ylabel("Peak frequency (Hz)")
    # ax2.set_ylim(0, max(1, max([max(p) if p else 0 for p in all_peak_freqs]) + 5))
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(0, T)
    ax2.grid(True)

    # Store scatter history
    peak_times = []
    peak_values = []

    # ---------------------------
    # AX3 — Contact History
    # ---------------------------
    line_contact, = ax3.plot([], [], lw=2)
    ax3.set_title("Contact History")
    ax3.set_ylabel("Contact (1) or No Contact (0)")
    ax3.set_xlabel("Time step")
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True)
    ax3.set_xlim(0, T)

    # ---------------------------
    # INIT FUNCTION
    # ---------------------------
    def init():
        video_im.set_data(rgb_frames[0])
        line_force.set_data([], [])
        line_contact.set_data([], [])

        peak_times.clear()
        peak_values.clear()
        scatter.set_offsets(np.empty((0, 2)))  # FIXED

        return video_im, line_force, scatter, line_contact

    # ---------------------------
    # ANIMATION STEP
    # ---------------------------
    def animate(i):
        # --- RGB video update ---
        video_im.set_data(rgb_frames[i])

        # --- Force ---
        line_force.set_data(range(i + 1), force_history[:i + 1])

        # --- Peak frequencies ---
        for f in all_peak_freqs[i]:
            peak_times.append(i)
            peak_values.append(f)

        if len(peak_times) > 0:
            scatter.set_offsets(np.column_stack([peak_times, peak_values]))
        else:
            scatter.set_offsets(np.empty((0, 2)))

        # --- Contact ---
        line_contact.set_data(range(i + 1), contact_history[:i + 1])

        return video_im, line_force, scatter, line_contact

    # ---------------------------
    # SAVE ANIMATION
    # ---------------------------
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=T,
        interval=1000 / fps,
        blit=True
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="mpeg4",
        extra_args=["-vcodec", "mpeg4", "-qscale", "5"]
    )

    ani.save(output_video_path, writer=writer)
    plt.close(fig)

