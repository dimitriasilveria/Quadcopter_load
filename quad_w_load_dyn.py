import numpy as np 
from utils import R3_so3
from scipy.linalg import expm
import matplotlib.pyplot as plt
from icecream import ic
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class quad_w_load_dyn:
    def __init__(self, mass_quad=0.835, mass_load=0.088, length=0.5, gravity=9.81):
        self.mq = mass_quad  # mass of the quadcopter
        self.ml = mass_load  # mass of the load
        self.l = length      # length of the cable
        self.g = gravity     # gravitational acceleration
        self.n_states = 15  # number of states in the system
        self.x = np.zeros((self.n_states, 1))  # state vector initialization
        self.x[6:9] = np.array([[0],[0],[1]])  # initial unit vector from load to quadcopter
        self.R = np.eye(3)  # rotation matrix initialization
        #0:3 position of load
        #3:6 velocity of load
        #6:9 unit vector from load to quadcopter
        #9:12 angular velocity of the load
        #12:15 angular velocity of the quadcopter
        #self.R represents the orientation of the quadcopter
        self.J_quad = 1e-3*np.diag([2.32, 2.32, 4])  # inertia matrix of the quadcopter
        self.e_3 = np.array([[0],[0],[1]])  # unit vector in z-direction
        self.dt = 0.01  # time step for integration
        self.h = 0.001 # runge-kutta sub-step size
        self.artists = []  # for animation

    # def x_l_dot(self):
    #     """Compute the time derivative of the load position."""
    #     return self.x[3:6]
    
    def v_l_dot(self,p, p_dot, f):
        """Compute the time derivative of the load velocity."""
        aux_1 = p.T@(f*self.R @ self.e_3)
        aux_2 = self.mq*self.l*p_dot.T@p_dot
        v_l_dot = (aux_1 - aux_2)*p/(self.ml+self.mq) - self.g*self.e_3
        return v_l_dot
    
    def p_dot(self,p, omega_l):
        """Compute the time derivative of the unit vector from load to quadcopter."""
        p_dot = np.cross(omega_l, p, axis=0)
        return p_dot
    
    def omega_l_dot(self,p, f):
        """Compute the time derivative of the load's angular velocity."""
        aux = f*self.R @ self.e_3
        if self.ml==0:
            omega_l_dot = np.zeros((3,1))
            return omega_l_dot
        omega_l_dot = (np.cross(-p, aux, axis=0))/(self.ml*self.l) 
        return omega_l_dot

    def R_quad_dot(self, omega):
        """Compute the time derivative of the rotation matrix."""
        omega_hat = R3_so3(omega)
        R_dot = omega_hat @ self.R
        return R_dot
    
    def omega_quad_dot(self, omega, tau):
        """Compute the time derivative of the quadcopter's angular velocity."""
        omega_quad_dot = np.linalg.inv(self.J_quad) @ (tau - np.cross(omega, self.J_quad @ omega, axis=0))
        return omega_quad_dot
    
    def dynamics(self,x, f, tau):
        """Compute the time derivative of the full state."""
        omega_l = x[9:12]
        omega_quad = x[12:15]
        p = x[6:9]
        v_l = x[3:6]
        p_dot = self.p_dot(p, omega_l)
        x_dot = np.zeros((self.n_states, 1))
        x_dot[0:3] = v_l
        x_dot[3:6] = self.v_l_dot(p, p_dot, f)
        x_dot[6:9] = p_dot
        x_dot[9:12] = self.omega_l_dot(p, f)
        R_dot = self.R_quad_dot(omega_quad)
        x_dot[12:15] = self.omega_quad_dot(omega_quad, tau)
        return x_dot, R_dot
    def runge_kutta_step(self, x0, f, tau):
        """Perform 4th order integration"""
        n = int(self.dt / self.h)
        for _ in range(n):
            k1_x, k1_R = self.dynamics(x0, f, tau)
            k2_x, k2_R = self.dynamics(x0 + 0.5 * self.h * k1_x, f, tau)
            k3_x, k3_R = self.dynamics(x0 + 0.5 * self.h * k2_x, f, tau)
            k4_x, k4_R = self.dynamics(x0 + self.h * k3_x, f, tau)

            x0 += (self.h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            self.R += (self.h / 6) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)
            self.R = self.R / np.linalg.norm(self.R, axis=0)  # re-orthonormalize R
            # self.R = self.R @ expm(self.h*R3_so3(x0[12:15]))
        self.x = x0

        return x0, self.R
    
    def quad_position(self):
        """Compute the position of the quadcopter."""
        p = self.x[6:9]
        x_l = self.x[0:3]
        x_q = x_l + self.l * p
        return x_q
    
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
    

if __name__ == "__main__":
    quad = quad_w_load_dyn()
    f = 9.81*(quad.mq + quad.ml)+2  # thrust force
    tau = np.array([[0.1],[0.1],[0]])
    N = 200
    t = np.linspace(0, N*quad.dt, N)
    Tau = np.array([np.zeros(N), np.sin(0.01*t), np.zeros(N)])
    f = 9.81*(quad.mq + quad.ml)  # thrust force
    Rot = np.zeros((3,3,N))
    X = np.zeros((6, N))

    x0 = np.zeros((quad.n_states, 1))
    X[0:3,0] = x0[0:3,0]
    
    # quad.x[6:9] = x0[6:9] = np.array([[1],[1],[1]])/np.linalg.norm(np.array([[1],[1],[1]]))
    X[3:6,0] = quad.quad_position().flatten()


    for i in range(N):
        if i == 0:
            f = 9.81*(quad.mq + quad.ml)  # thrust force
            # tau = np.array([[0],[-0.01],[0]])
        else: 
            f = 9.81*(quad.mq + quad.ml)  # thrust force
            tau = np.array([[0],[0.1],[0]])
        x, R_ = quad.runge_kutta_step(quad.x, f, tau)
        X[0:3,i] = x[0:3,0]
        X[3:6,i] = quad.quad_position().flatten()
        Rot[:,:,i] = R_

    plot_quadcopter_pendulum_3d(X, Rot=Rot, cable_length=quad.l, indices=[0, N//2, N-1])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(X[0,:], X[1,:], X[2,:], label='Load Trajectory')
    # ax.plot(X[3,:], X[4,:], X[5,:], label='Quadcopter Trajectory')
    # # quad.draw_pendulum_3d(ax, X[3:6,-1], X[0:3,-1])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()

    # quad.animate_quad_pendulum_3d(X[3:6,:], X[0:3,:], Rot=Rot, interval=50, elev=20, azim=-60)