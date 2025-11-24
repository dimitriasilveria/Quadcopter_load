from icecream import ic
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from scipy.linalg import logm

def R3_so3(w):
    v3 = w[2,0]
    v2 = w[1,0]
    v1 = w[0,0]
    so3 = np.array([[ 0 , -v3,  v2],
          [v3,   0, -v1],
          [-v2,  v1,   0]])

    return so3

def R2_so2(w):
    v2 = w[1,0]
    v1 = w[0,0]
    so2 = np.array([[ 0 , -v2],
          [v2,   0]])

    return so2

def skew_to_R3(v):
    w1 = v[2,1]
    w2 = v[0,2]
    w3 = v[1,0]
    w = np.array([w1,w2,w3]).reshape((3,1))
    return w

def so3_R3(Rot):

    log_R = logm(Rot)
    w1 = log_R[2,1]
    w2 = log_R[0,2]
    w3 = log_R[1,0]
    w = np.array([w1,w2,w3]).reshape((3,1))
    return w

def calc_w_from_Rdot(Rot, Rot_prev, dt):
    w = so3_R3(logm(Rot_prev.T @ Rot)) / dt
    w = w.reshape((3,1))
    return w

def _set_axes_equal(ax):
    """Make 3D axes have equal scale (works for matplotlib 3D axes)."""
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

def animate_quad_and_load(x_load, x_quad, R=None, x_des=None,
                          quad_arm_length=0.12, trail=50,
                          interval=40, figsize=(9,7), elev=20, azim=-60,
                          show=True):
    """
    Simple 3D animation of quad + suspended load with static desired trajectory.

    Parameters
    ----------
    x_load : (3, N) ndarray
        Load positions (columns are time samples).
    x_quad : (3, N) ndarray
        Quadcopter positions (columns are time samples).
    R : (3,3,N) ndarray or None
        Rotation matrices for the quad at each frame (optional).
    x_des : (3, M) or (6, M) ndarray or None
        Desired trajectory. If (3,M) treated as desired load positions.
        If (6,M) rows 0:3=desired load, 3:6=desired quad.
        The whole desired path is drawn once (dashed line) and remains visible.
    quad_arm_length : float
        Visual length of the quad arms.
    trail : int
        Number of past samples to draw as a trail (set 0 to disable).
    interval : int
        Milliseconds between frames.
    Returns
    -------
    fig, ax, anim
    """
    # validate and normalize inputs
    x_load = np.asarray(x_load)
    x_quad = np.asarray(x_quad)
    assert x_load.shape[0] == 3 and x_quad.shape[0] == 3, "x_load and x_quad must be shape (3, N)"
    N = x_load.shape[1]
    if x_quad.shape[1] != N:
        raise ValueError("x_load and x_quad must have same number of columns")

    # desired trajectory processing (no moving marker; just plot full path)
    load_des = quad_des = None
    if x_des is not None:
        x_des = np.asarray(x_des)
        if x_des.ndim != 2 or x_des.shape[0] not in (3, 6):
            raise ValueError("x_des must be shape (3,M) or (6,M)")
        if x_des.shape[0] == 6:
            load_des = x_des[0:3, :].T
            quad_des = x_des[3:6, :].T
        else:
            load_des = x_des.T

    # prepare figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)

    # compute bounds including desired if present
    all_pts = np.vstack((x_load.T, x_quad.T))
    if load_des is not None:
        all_pts = np.vstack((all_pts, load_des))
    if quad_des is not None:
        all_pts = np.vstack((all_pts, quad_des))
    if all_pts.size:
        pad = 0.1 * np.max(np.ptp(all_pts, axis=0))
        mins = np.min(all_pts, axis=0) - pad
        maxs = np.max(all_pts, axis=0) + pad
        # ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])

    #static (faint) full actual trajectories
    ax.plot(x_load[0,:], x_load[1,:], x_load[2,:], linewidth=1.5, alpha=0.2, label='load traj')
    ax.plot(x_quad[0,:], x_quad[1,:], x_quad[2,:], linestyle='--', linewidth=1.0, alpha=0.2, label='quad traj')

    # static desired (always visible, dashed)
    if load_des is not None:
        ax.plot(load_des[:,0], load_des[:,1], load_des[:,2], linestyle=':', linewidth=1.6, alpha=0.95, label='desired load')
    if quad_des is not None:
        ax.plot(quad_des[:,0], quad_des[:,1], quad_des[:,2], linestyle=':', linewidth=1.6, alpha=0.95, label='desired quad')

    # dynamic artists
    load_marker, = ax.plot([], [], [], marker='o', markersize=6, linestyle='None', label='load')
    quad_marker, = ax.plot([], [], [], marker='^', markersize=8, linestyle='None', label='quad')
    pendulum_line, = ax.plot([], [], [], linewidth=2, color='k')
    arm1_line, = ax.plot([], [], [], linewidth=3)
    arm2_line, = ax.plot([], [], [], linewidth=3)

    if trail > 0:
        load_trail_line, = ax.plot([], [], [], linewidth=2, alpha=0.9)
        quad_trail_line, = ax.plot([], [], [], linewidth=1.5, alpha=0.9)
    else:
        load_trail_line = quad_trail_line = None

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.legend()
    _set_axes_equal(ax)

    def _set_arm_lines(center, Rmat):
        if Rmat is None:
            p1 = center + np.array([ quad_arm_length, 0.0, 0.0])
            p2 = center - np.array([ quad_arm_length, 0.0, 0.0])
            p3 = center + np.array([0.0,  quad_arm_length, 0.0])
            p4 = center - np.array([0.0,  quad_arm_length, 0.0])
        else:
            x_axis = Rmat[:,0].flatten()
            y_axis = Rmat[:,1].flatten()
            p1 = center + quad_arm_length * x_axis
            p2 = center - quad_arm_length * x_axis
            p3 = center + quad_arm_length * y_axis
            p4 = center - quad_arm_length * y_axis
        return (p1,p2,p3,p4)

    def init():
        pendulum_line.set_data([], []); pendulum_line.set_3d_properties([])
        arm1_line.set_data([], []); arm1_line.set_3d_properties([])
        arm2_line.set_data([], []); arm2_line.set_3d_properties([])
        load_marker.set_data([], []); load_marker.set_3d_properties([])
        quad_marker.set_data([], []); quad_marker.set_3d_properties([])
        if load_trail_line is not None:
            load_trail_line.set_data([], []); load_trail_line.set_3d_properties([])
            quad_trail_line.set_data([], []); quad_trail_line.set_3d_properties([])
        return (pendulum_line, arm1_line, arm2_line, load_marker, quad_marker, load_trail_line, quad_trail_line)

    def update(i):
        i = int(i) % N
        load = x_load[:, i]
        quad = x_quad[:, i]

        # markers
        load_marker.set_data([load[0]], [load[1]]); load_marker.set_3d_properties([load[2]])
        quad_marker.set_data([quad[0]], [quad[1]]); quad_marker.set_3d_properties([quad[2]])

        # pendulum
        pendulum_line.set_data([load[0], quad[0]], [load[1], quad[1]])
        pendulum_line.set_3d_properties([load[2], quad[2]])

        # arms
        if R is not None and R.shape == (3,3,N):
            Rm = R[:,:,i]
        else:
            Rm = None
        p1,p2,p3,p4 = _set_arm_lines(quad, Rm)
        arm1_line.set_data([p1[0], p2[0]], [p1[1], p2[1]]); arm1_line.set_3d_properties([p1[2], p2[2]])
        arm2_line.set_data([p3[0], p4[0]], [p3[1], p4[1]]); arm2_line.set_3d_properties([p3[2], p4[2]])

        # trails
        if trail > 0:
            start = max(0, i - trail)
            load_seg = x_load[:, start:i+1].T
            quad_seg = x_quad[:, start:i+1].T
            load_trail_line.set_data(load_seg[:,0], load_seg[:,1]); load_trail_line.set_3d_properties(load_seg[:,2])
            quad_trail_line.set_data(quad_seg[:,0], quad_seg[:,1]); quad_trail_line.set_3d_properties(quad_seg[:,2])

        return (pendulum_line, arm1_line, arm2_line, load_marker, quad_marker, load_trail_line, quad_trail_line)

    anim = FuncAnimation(fig, update, frames=np.arange(0, N), init_func=init, interval=interval, blit=False)

    if show:
        plt.show()

    return fig, ax, anim