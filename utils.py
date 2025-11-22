import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def R3_so3(w):
    v3 = w[2,0]
    v2 = w[1,0]
    v1 = w[0,0]
    so3 = np.array([[ 0 , -v3,  v2],
          [v3,   0, -v1],
          [-v2,  v1,   0]])

    return so3

def so3_R3(Rot):

    log_R = logm(Rot)
    w1 = log_R[2,1]
    w2 = log_R[0,2]
    w3 = log_R[1,0]
    w = np.array([w1,w2,w3]).T
    return w

def _set_axes_equal(ax):
    """
    Make 3D axes have equal scale.

    This is a helper for matplotlib 3D. It modifies axis limits so that the data
    is not distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_quadcopter_pendulum_3d(X, Rot=None, cable_length=None, indices=None,
                                quad_arm_length=0.12, show=True, figsize=(9,7)):
    """
    Plot 3D trajectory and draw quadcopter + pendulum at specified indices.

    Parameters
    ----------
    X : ndarray (6, N)
        State snapshot array where:
          X[0:3, i] = load position (x_l)
          X[3:6, i] = quad position (x_q)
    Rot : ndarray (3,3,N) or None
        Rotation matrices of the quadcopter for each time step. If None or
        zeros, the quad will be drawn aligned with world axes.
    cable_length : float or None
        Cable length (optional). If provided, will be used to mark the nominal
        pendulum length (not required to draw actual pendulum line).
    indices : sequence of ints or None
        Indices at which to draw the quad + pendulum. If None uses
        [0, N//2, N-1].
    quad_arm_length : float
        Length of the two orthogonal arms drawn for the quadcopter.
    show : bool
        If True, calls plt.show().
    figsize : tuple
        Figure size.
    """
    assert X.ndim == 2 and X.shape[0] >= 6, "X must be shape (6, N)"
    N = X.shape[1]

    if indices is None:
        indices = [0, N//2, N-1]
    else:
        # keep within bounds and unique
        indices = sorted({max(0, min(int(i), N-1)) for i in indices})

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)

    # plot trajectories: load (solid) and quad (dashed)
    load_pos = X[0:3, :].T  # (N,3)
    quad_pos = X[3:6, :].T  # (N,3)
    ax.plot(load_pos[:,0], load_pos[:,1], load_pos[:,2], label='load trajectory', linewidth=2)
    ax.plot(quad_pos[:,0], quad_pos[:,1], quad_pos[:,2], '--', label='quad trajectory', linewidth=1.5)

    # draw markers at indices
    colors = ['C0','C1','C2']  # distinct colors for start/mid/end
    labels = ['start','middle','end']
    for ii, idx in enumerate(indices):
        c = colors[ii % len(colors)]
        label = labels[ii] if ii < len(labels) else f'idx {idx}'
        # marker for load and quad
        ax.scatter(load_pos[idx,0], load_pos[idx,1], load_pos[idx,2], color=c, s=40, marker='o')
        ax.scatter(quad_pos[idx,0], quad_pos[idx,1], quad_pos[idx,2], color=c, s=40, marker='^')
        # draw pendulum (line between load and quad)
        xs = [load_pos[idx,0], quad_pos[idx,0]]
        ys = [load_pos[idx,1], quad_pos[idx,1]]
        zs = [load_pos[idx,2], quad_pos[idx,2]]
        ax.plot(xs, ys, zs, '-', color=c, linewidth=2, alpha=0.9)

        # draw quad arms using rotation matrix if provided
        if (Rot is not None) and (Rot.shape == (3,3,N)):
            R = Rot[:,:,idx]
            # R columns are body axes (assuming R maps body->inertial)
            x_axis = R[:,0].flatten()
            y_axis = R[:,1].flatten()
            center = quad_pos[idx,:]
            # arm endpoints
            p1 = center + quad_arm_length * x_axis
            p2 = center - quad_arm_length * x_axis
            p3 = center + quad_arm_length * y_axis
            p4 = center - quad_arm_length * y_axis
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, linewidth=3)
            ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=c, linewidth=3)
        else:
            # fallback: draw arms aligned with world XY
            center = quad_pos[idx,:]
            ax.plot([center[0]-quad_arm_length, center[0]+quad_arm_length],
                    [center[1], center[1]], [center[2], center[2]],
                    color=c, linewidth=3)
            ax.plot([center[0], center[0]],
                    [center[1]-quad_arm_length, center[1]+quad_arm_length],
                    [center[2], center[2]],
                    color=c, linewidth=3)

        # annotate
        ax.text(quad_pos[idx,0], quad_pos[idx,1], quad_pos[idx,2],
                f'  {label}', color=c)

    # optionally show nominal cable length as a transparent sphere (if provided)
    if cable_length is not None:
        # plot a sphere centered at each quad sample with radius cable_length (outline)
        u = np.linspace(0, 2*np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        for idx in indices:
            cx, cy, cz = quad_pos[idx,:]
            xs = cx + cable_length * np.outer(np.cos(u), np.sin(v))
            ys = cy + cable_length * np.outer(np.sin(u), np.sin(v))
            zs = cz + cable_length * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, alpha=0.08)

    # labels + legend
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()

    # nice equal aspect
    _set_axes_equal(ax)

    if show:
        plt.show()

    return fig, ax

def _set_axes_equal_3d(ax):
    """Make 3D axes have equal scale (works with pre-known limits)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def animate_quadcopter_pendulum_3d(X, Rot=None, cable_length=None,
                                   quad_arm_length=0.12, interval=40,
                                   trail=50, figsize=(9,7), elev=20, azim=-60,
                                   show=True):
    """
    Create and return a matplotlib.animation.FuncAnimation showing:
      - Load trajectory (solid)
      - Quad trajectory (dashed)
      - Current load (dot) and quad (triangle) markers
      - Pendulum line between load and quad
      - Two quad arms (using Rot if provided)
      - Optional short trailing path (trail param)

    Parameters
    ----------
    X : ndarray (6, N)
        State snapshots: rows 0:3 = load pos, rows 3:6 = quad pos
    Rot : ndarray (3,3,N) or None
        Rotation matrices for the quad (optional). If None, arms are world-aligned.
    cable_length : float or None
        If provided, draws a faint sphere around quad at selected times (not animated)
    quad_arm_length : float
        Length of the quad arms drawn
    interval : int
        Milliseconds between frames
    trail : int
        Number of past samples to show as trail for each object (set 0 to disable)
    figsize, elev, azim : plotting camera setup
    show : bool
        If True, calls plt.show() before returning (useful in scripts)
    Returns
    -------
    anim : FuncAnimation
        The animation object (keeps a reference to prevent GC). Also returns fig, ax.
    """
    assert X.ndim == 2 and X.shape[0] >= 6, "X must be shape (6, N)"
    N = X.shape[1]

    load_traj = X[0:3, :].T  # (N,3)
    quad_traj = X[3:6, :].T  # (N,3)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)

    # precompute global bounds and set them so the axes don't jump
    all_pts = np.vstack((load_traj, quad_traj))
    pad = 0.1 * np.max(np.ptp(all_pts, axis=0))
    xmin, ymin, zmin = all_pts.min(axis=0) - pad
    xmax, ymax, zmax = all_pts.max(axis=0) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    _set_axes_equal_3d(ax)

    # static full trajectories (faint)
    traj_load_line, = ax.plot(load_traj[:,0], load_traj[:,1], load_traj[:,2],
                              linewidth=1.5, alpha=0.25, label='load trajectory')
    traj_quad_line, = ax.plot(quad_traj[:,0], quad_traj[:,1], quad_traj[:,2],
                              linestyle='--', linewidth=1.0, alpha=0.25, label='quad trajectory')

    # dynamic artists for current frame
    load_point, = ax.plot([load_traj[0,0]], [load_traj[0,1]], [load_traj[0,2]],
                          marker='o', markersize=6, linestyle='None')
    quad_point, = ax.plot([quad_traj[0,0]], [quad_traj[0,1]], [quad_traj[0,2]],
                          marker='^', markersize=8, linestyle='None')

    pendulum_line, = ax.plot([], [], [], linewidth=2)

    # quad arms (two lines)
    arm1_line, = ax.plot([], [], [], linewidth=3)
    arm2_line, = ax.plot([], [], [], linewidth=3)

    # optional trailing path lines
    if trail > 0:
        load_trail_line, = ax.plot([], [], [], linewidth=2, alpha=0.9)
        quad_trail_line, = ax.plot([], [], [], linewidth=1.5, alpha=0.9)
    else:
        load_trail_line = quad_trail_line = None

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()

    # helper to extract axis data
    def _set_arm_lines(center, Rmat):
        """Return endpoints for two orthogonal arms given center and (optional) Rmat."""
        if Rmat is None:
            # world-aligned arms in XY-plane
            p1 = center + np.array([ quad_arm_length, 0.0, 0.0])
            p2 = center + np.array([-quad_arm_length, 0.0, 0.0])
            p3 = center + np.array([0.0,  quad_arm_length, 0.0])
            p4 = center + np.array([0.0, -quad_arm_length, 0.0])
        else:
            # assume Rmat maps body->inertial and columns are body axes
            x_axis = Rmat[:,0].flatten()
            y_axis = Rmat[:,1].flatten()
            p1 = center + quad_arm_length * x_axis
            p2 = center - quad_arm_length * x_axis
            p3 = center + quad_arm_length * y_axis
            p4 = center - quad_arm_length * y_axis
        return (p1,p2,p3,p4)

    # init function
    def init():
        pendulum_line.set_data([], [])
        pendulum_line.set_3d_properties([])
        arm1_line.set_data([], [])
        arm1_line.set_3d_properties([])
        arm2_line.set_data([], [])
        arm2_line.set_3d_properties([])
        if load_trail_line is not None:
            load_trail_line.set_data([], []); load_trail_line.set_3d_properties([])
            quad_trail_line.set_data([], []); quad_trail_line.set_3d_properties([])
        load_point.set_data([], []); load_point.set_3d_properties([])
        quad_point.set_data([], []); quad_point.set_3d_properties([])
        return (pendulum_line, arm1_line, arm2_line, load_point, quad_point,
                load_trail_line, quad_trail_line)

    # animation update for frame i
    def update(i):
        # clamp i
        i = int(i) % N
        load = load_traj[i]
        quad = quad_traj[i]

        # update markers
        load_point.set_data([load[0]], [load[1]])
        load_point.set_3d_properties([load[2]])
        quad_point.set_data([quad[0]], [quad[1]])
        quad_point.set_3d_properties([quad[2]])

        # pendulum line
        xs = [load[0], quad[0]]
        ys = [load[1], quad[1]]
        zs = [load[2], quad[2]]
        pendulum_line.set_data(xs, ys)
        pendulum_line.set_3d_properties(zs)

        # arms
        if Rot is not None and Rot.shape == (3,3,N):
            Rm = Rot[:,:,i]
        else:
            Rm = None
        p1, p2, p3, p4 = _set_arm_lines(quad, Rm)
        arm1_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        arm1_line.set_3d_properties([p1[2], p2[2]])
        arm2_line.set_data([p3[0], p4[0]], [p3[1], p4[1]])
        arm2_line.set_3d_properties([p3[2], p4[2]])

        # trails
        if trail > 0:
            start = max(0, i - trail)
            load_seg = load_traj[start:i+1]
            quad_seg = quad_traj[start:i+1]
            load_trail_line.set_data(load_seg[:,0], load_seg[:,1])
            load_trail_line.set_3d_properties(load_seg[:,2])
            quad_trail_line.set_data(quad_seg[:,0], quad_seg[:,1])
            quad_trail_line.set_3d_properties(quad_seg[:,2])

        return (pendulum_line, arm1_line, arm2_line, load_point, quad_point,
                load_trail_line, quad_trail_line)

    anim = FuncAnimation(fig, update, frames=np.arange(0, N),
                         init_func=init, interval=interval, blit=False)

    # optionally draw a faint sphere at a few indices to show cable_length (not animated)
    if cable_length is not None:
        try:
            # draw at start, middle, end (faint)
            idxs = [0, N//2, N-1]
            u = np.linspace(0, 2*np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            for idx in idxs:
                cx, cy, cz = quad_traj[idx]
                xs = cx + cable_length * np.outer(np.cos(u), np.sin(v))
                ys = cy + cable_length * np.outer(np.sin(u), np.sin(v))
                zs = cz + cable_length * np.outer(np.ones_like(u), np.cos(v))
                ax.plot_wireframe(xs, ys, zs, alpha=0.06)
        except Exception:
            pass

    if show:
        plt.show()

    return fig, ax, anim   

def plot_desired_trajectory_on_ax(ax, X_des, label_prefix="desired", alpha=0.9, linewidth=1.5):
    """
    Overlay desired trajectory on an existing 3D axis `ax`.
    X_des: array with shape (6,N) or (3,N). If (6,N), rows 0:3 = load, 3:6 = quad.
    """
    assert X_des.ndim == 2 and X_des.shape[0] in (3,6), "X_des must be (3,N) or (6,N)"
    if X_des.shape[0] == 6:
        load_des = X_des[0:3, :].T
        quad_des = X_des[3:6, :].T
    else:
        # assume this is quad desired positions
        load_des = None
        quad_des = X_des.T

    if load_des is not None:
        ax.plot(load_des[:,0], load_des[:,1], load_des[:,2],
                linestyle=':', linewidth=linewidth, alpha=alpha, label=f'{label_prefix} load')

    ax.plot(quad_des[:,0], quad_des[:,1], quad_des[:,2],
            linestyle=':', linewidth=linewidth, alpha=alpha, label=f'{label_prefix} quad')

    # mark start/goal
    ax.scatter(quad_des[0,0], quad_des[0,1], quad_des[0,2], marker='x', s=40, label=f'{label_prefix} start')
    ax.scatter(quad_des[-1,0], quad_des[-1,1], quad_des[-1,2], marker='*', s=60, label=f'{label_prefix} goal')

    # update legend (caller can call ax.legend() afterwards)
    return ax


def animate_quadcopter_pendulum_with_desired(X, Rot=None, X_des=None, cable_length=None,
                                             quad_arm_length=0.12, interval=40, trail=50,
                                             figsize=(9,7), elev=20, azim=-60, show=True):
    """
    Wrapper around animate_quadcopter_pendulum_3d that also shows a desired trajectory.
    - X: actual data in your (6,N) format (rows 0:3 load, 3:6 quad)
    - X_des: desired trajectory (6,N) or (3,N). If None, nothing added.
    Returns: fig, ax, anim (same as animate_quadcopter_pendulum_3d)
    """
    # call your existing animation creator but disable show so we can modify the axis before presenting
    fig, ax, anim = animate_quadcopter_pendulum_3d(X, Rot=Rot, cable_length=cable_length,
                                                  quad_arm_length=quad_arm_length, interval=interval,
                                                  trail=trail, figsize=figsize, elev=elev, azim=azim, show=False)

    # overlay desired
    if X_des is not None:
        # normalize shapes
        X_des = np.asarray(X_des)
        if X_des.ndim != 2 or X_des.shape[0] not in (3,6):
            raise ValueError("X_des must be shape (6,N) or (3,N)")

        # If desired length differs from actual N, resample or plot as-is.
        # Here we just plot as-is; if you want resampling, tell me and I'll add it.
        try:
            plot_desired_trajectory_on_ax(ax, X_des, label_prefix="desired", alpha=0.9, linewidth=1.8)
        except Exception as e:
            # don't crash animation; print helpful message
            print("Could not plot desired trajectory:", e)

    ax.legend()
    _set_axes_equal(ax)  # keep aspect equal (re-use your helper)

    if show:
        plt.show()

    return fig, ax, anim