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
    X, Y = np.meshgrid(x, y, indexing='ij')
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
        U_frames[i] = U_plot

    # Create a 3D surface animation for 480p-ish output
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Determine fixed z-limits across all frames so the vertical scale remains constant
    zmin = float(np.min(U_frames))
    zmax = float(np.max(U_frames))

    # initial surface: single color (no color gradient)
    surf = ax.plot_surface(X, Y, U_frames[0], color='C0', linewidth=0, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_zlim(zmin, zmax)
    title = ax.set_title(f'Solution at t = {times[0]:.4f}')

    surf_holder = [surf]

    def update(frame):
        try:
            surf_holder[0].remove()
        except Exception:
            pass
        surf_new = ax.plot_surface(X, Y, U_frames[frame], color='C0', linewidth=0, antialiased=True)
        title.set_text(f'Solution at t = {times[frame]:.4f}')
        ax.set_zlim(zmin, zmax)
        surf_holder[0] = surf_new
        return surf_new,

    anim = FuncAnimation(fig, update, frames=num_frames, blit=False)
    print(f"Saving animation to {filename} (this may take a while)...")
    from matplotlib import animation
    writers = animation.writers
    if writers.is_available('ffmpeg'):
        FFMpegWriter = animation.FFMpegWriter
        writer = FFMpegWriter(fps=15, bitrate=8000, extra_args=['-preset', 'ultrafast'])
        anim.save(filename, writer=writer)
        print("Animation saved.")
    else:
        from matplotlib.animation import PillowWriter
        gif_name = filename.rsplit('.', 1)[0] + '.gif'
        writer = PillowWriter(fps=10)
        anim.save(gif_name, writer=writer)
        print(f"ffmpeg not available; saved GIF to {gif_name}")
    plt.close(fig)



def plot_ns_solution_2d_vector(u_2d, v_2d, times, gll_pts, t, N_pts=40):
    """
    Plot a 2D velocity field (u, v) at time t.

    u_2d, v_2d : solution arrays in shape (N+1, N+1)
    """
    # pick nearest index
    t_idx = np.argmin(np.abs(times - t))

    # construct spectral interpolants
    u_func = construct_solution_2d_fast(u_2d, gll_pts)
    v_func = construct_solution_2d_fast(v_2d, gll_pts)

    # grid for plotting
    x = np.linspace(-1, 1, N_pts)
    y = np.linspace(-1, 1, N_pts)
    # X, Y = np.meshgrid(x, y)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # evaluate velocity field
    U = u_func(x, y)
    V = v_func(x, y)
    
    # transpose if needed
    if U.shape != X.shape:
        # U = U.T
        # V = V.T
        print("SHAPE MISMATCH:", U.shape, X.shape)

    # magnitude
    speed = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(6,5))

    # background magnitude color plot
    c = ax.pcolormesh(X, Y, speed, shading='auto')

    # quiver (vector arrows)
    ax.quiver(X, Y, U/speed, V/speed, color='white', scale=40)

    fig.colorbar(c, ax=ax, label='|v|')
    ax.set_title(f"Velocity field at t = {t:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

def plot_pressure_2d(p1d, times, gll_pts, t, N_pts=50):
    """
    Plot the 2D pressure field reconstructed from a 1D spectral-element vector.

    Parameters
    ----------
    p1d : ndarray
        Pressure values in a flattened (N+1)^2 array at GLL points.
    times : ndarray
        Array of simulation times.
    gll_pts : ndarray
        1D GLL nodes used for interpolation.
    t : float
        Desired physical time to plot.
    N_pts : int
        Number of uniform grid points per dimension for plotting.
    """

    # find closest time index
    t_idx = np.argmin(np.abs(times - t))

    # rebuild interpolation function
    p_func = construct_solution_2d_fast(p1d, gll_pts)

    # uniform grid
    x = np.linspace(-1, 1, N_pts)
    y = np.linspace(-1, 1, N_pts)
    X, Y = np.meshgrid(x, y)

    # evaluate
    P_plot = p_func(x, y)
    if P_plot.shape != X.shape:
        P_plot = P_plot.T

    # plotting
    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, P_plot, shading='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Pressure at t = {t:.3f}')
    fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.show()