import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lagrange_utils import construct_solution_2d_fast


def plot_solution_2d(U_solution, times, gll_pts, t, N_pts=50):
    """
    Plots the solution at a given time t as a color plot.
    """
    t_idx = np.argmin(np.abs(times - t))
    u_func = construct_solution_2d_fast(U_solution[t_idx, :, :], gll_pts)
    
    x = np.linspace(-1, 1, N_pts)
    y = np.linspace(-1, 1, N_pts)
    X, Y = np.meshgrid(x, y)
    # Pass 1D arrays to u_func, then transpose for plotting if needed
    U_plot = u_func(x, y)
    if U_plot.shape != X.shape:
        U_plot = U_plot.T  # Ensure shape matches meshgrid

    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, U_plot, cmap='viridis', shading='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Solution at t = {times[t_idx]:.2f}')
    fig.colorbar(c, ax=ax)
    plt.show()

def animate_solution_2d(U_solution, times, gll_pts, filename, N_pts=100):
    """
    Creates a color plot animation of the solution evolving in time.
    Optimized for performance and outputs an MP4 file.
    The colorbar (z-axis) is fixed between 0 and 1.
    Now avoids updating the colorbar and uses ffmpeg ultrafast preset with 480p video resolution.
    """
    x = np.linspace(-1, 1, N_pts)
    y = np.linspace(-1, 1, N_pts)
    X, Y = np.meshgrid(x, y)

    # Precompute all frames with progress printout
    num_frames = len(times)
    U_frames = np.empty((num_frames, N_pts, N_pts))
    print(f"Precomputing {num_frames} frames at {N_pts}x{N_pts} resolution...")
    for i in range(num_frames):
        if (i % max(1, num_frames // 10) == 0) or (i == num_frames - 1):
            print(f"  Frame {i+1}/{num_frames} ({100*(i+1)//num_frames}%)")
        u_func = construct_solution_2d_fast(U_solution[i, :, :], gll_pts)
        U_plot = u_func(x, y)
        if U_plot.shape != X.shape:
            U_plot = U_plot.T
        U_frames[i] = U_plot

    # Set up the figure for 480p resolution
    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    c = ax.pcolormesh(X, Y, U_frames[0], cmap='viridis', shading='auto', vmin=-1, vmax=1)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    title = ax.set_title(f'Solution at t = {times[0]:.4f}')

    def update(frame):
        U_plot = U_frames[frame]
        c.set_array(U_plot.ravel())
        title.set_text(f'Solution at t = {times[frame]:.4f}')
        return c, title

    anim = FuncAnimation(fig, update, frames=num_frames, blit=True)
    print(f"Saving animation to {filename} (this may take a while)...")
    # Use ffmpeg ultrafast preset for faster saving
    anim.save(filename, writer='ffmpeg', fps=15, bitrate=8000, extra_args=['-preset', 'ultrafast'])
    print("Animation saved.")
    plt.close(fig)