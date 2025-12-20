import numpy as np 
from utils import R2_so2, animate_quad_and_load
from scipy.linalg import expm
import matplotlib.pyplot as plt
from icecream import ic
from matplotlib.animation import FuncAnimation
import matplotlib
# matplotlib.use('TkAgg')   # or 'Qt5Agg' if you have Qt
from mpl_toolkits.mplot3d import Axes3D
#import ode from scipy
from scipy.integrate import RK45
from IPython.display import HTML, display
import os

class quad_w_load_dyn:
    def __init__(self, mass_quad=0.835, mass_load=0.088, length=0.5, gravity=9.81):
        '''
        0:2 position of load
        2:4 velocity of load
        4 angle of the load
        5 angular velocity of the load
        6 angle of the quadcopter
        7 angular velocity of the quadcopter
        '''
        self.mq = mass_quad  # mass of the quadcopter
        self.ml = mass_load  # mass of the load
        self.l = length      # length of the cable
        self.L = 0.175  # distance from the center to each motor
        self.g = gravity     # gravitational acceleration
        self.n_states = 8  # number of states in the system
        self.x = np.zeros((self.n_states, 1))  # state vector initialization
        self.J_quad = 1e-3*np.diag([2.32, 4])  # inertia matrix of the quadcopter
        self.J_quad_inv = np.linalg.inv(self.J_quad)
        self.e_3 = np.array([[0],[1]])  # unit vector in z-direction
        self.dt = 0.01  # time step for integration
        self.h = 0.001 # runge-kutta sub-step size
        self.artists = []  # for animation
        self.km = 1.5e-9  # motor constant
        self.kf = 6.11e-8  # thrust constant
        self.L = 0.175  # distance from the center to each motor
        self.w_max = 7800  # maximum motor speed in RPM
        self.w_min = 1200     # minimum motor speed in RPM

    # def x_l_dot(self):
    #     """Compute the time derivative of the load position."""
    #     return self.x[3:6]

    def calc_max_torque_thrust(self):
        tau_x_max = self.L * self.kf * (self.w_max**2 - self.w_min**2)
        max_thrust = self.kf * 4 * self.w_max**2
        return max_thrust, tau_x_max
    
    def calc_min_torque_thrust(self):
        tau_x_min = self.L * self.kf * (self.w_min**2 - self.w_max**2)
        min_thrust = self.kf * 4 * self.w_min**2
        return min_thrust, tau_x_min
    
    def v_l_dot(self,w_l, phi_l, phi_q, f):
        """Compute the time derivative of the load velocity."""
        a_y = ((-self.mq*self.l*w_l**2-f*np.cos(phi_l-phi_q)*np.sin(phi_l)))/(self.ml+self.mq)
        a_z = ((self.mq*self.l*w_l**2 + f*np.cos(phi_l-phi_q)*np.cos(phi_l))/(self.ml+self.mq)) - self.g
        v_l_dot = np.array([[a_y],[a_z]])
        return v_l_dot
    
    def omega_l_dot(self, phi_l, phi_q, f):
        """Compute the time derivative of the load's angular velocity."""
        omega_l_dot = -(f*np.sin(phi_l-phi_q))/(self.l*self.mq)
        return omega_l_dot

    def omega_quad_dot(self, tau):
        """Compute the time derivative of the quadcopter's angular velocity."""
        omega_quad_dot = tau/self.J_quad[0,0]
        return omega_quad_dot
    
    #0:2 position of load
    #2:4 velocity of load
    #4 angle of the load
    #5 angular velocity of the load
    #6 angle of the quadcopter
    #7 angular velocity of the quadcopter
    def dynamics(self,x, f, tau):
        """Compute the time derivative of the full state."""
        x_dot = np.zeros((self.n_states,1))
        v_l = x[2:4,0]
        phi_l = x[4,0]
        w_l = x[5,0]
        phi_q = x[6,0]
        w_q = x[7,0]
        x_dot[0:2,0] = v_l
        x_dot[2:4,0] = self.v_l_dot(w_l, phi_l, phi_q, f).flatten()
        x_dot[4,0] = w_l
        x_dot[5,0] = self.omega_l_dot(phi_l, phi_q, f)
        x_dot[6,0] = w_q
        x_dot[7,0] = self.omega_quad_dot(tau)
        return x_dot
    def runge_kutta_step(self, x0, f, tau):
        """Perform 4th order integration"""
        t_span = (0, self.dt)
        def dyn(t, x):
            return self.dynamics(x.reshape((self.n_states,1)), f, tau).flatten()
        rk = RK45(dyn, t_span[0], x0.flatten(), t_span[1],rtol=1e-2, atol=1e-4)
        while rk.status == 'running':
            rk.step()
        sol = rk
        if sol.y.ndim  > 1:
            x0 = sol.y[:, -1].reshape((self.n_states,1))
        else:
            x0 = sol.y.reshape((self.n_states,1))

        self.x = x0
        phi_q = x0[6,0]
        self.R = np.array([[np.cos(phi_q), -np.sin(phi_q)],
                           [np.sin(phi_q),  np.cos(phi_q)]])
        return x0
    
    def quad_position(self):
        """Compute the position of the quadcopter."""
        x_l = self.x[0:2]
        x_q = x_l + self.l * np.array([[-np.sin(self.x[4,0])],[np.cos(self.x[4,0])]])
        return x_q
    


if __name__ == "__main__":
    quad = quad_w_load_dyn()
    print(quad.calc_max_torque_thrust(), quad.calc_min_torque_thrust())
    input()
    f = 9.81*(quad.mq + quad.ml)+2  # thrust force
    tau = -0.05
    N = 200
    t = np.linspace(0, N*quad.dt, N)
    Tau = np.array([np.zeros(N), np.sin(0.01*t), np.zeros(N)])
    f = 9.81*(quad.mq + quad.ml)  # thrust force
    Rot = np.zeros((2,2,N))
    quad.x[0:2] = np.array([[10],[10]])
    X = np.zeros((4, N))
    # Xfull: full-state history, shape (n_states, N)
    Xfull = np.zeros((quad.n_states, N))

    x0 = np.zeros((quad.n_states, 1))
    X[0:2,0] = x0[0:2,0]
    
    # quad.x[6:9] = x0[6:9] = np.array([[1],[1],[1]])/np.linalg.norm(np.array([[1],[1],[1]]))
    X[2:4,0] = quad.quad_position().flatten()

    for i in range(N):
        if i == 0:
            f = 9.81*(quad.mq + quad.ml)+2  # thrust force
            # tau = np.array([[0],[-0.01],[0]])
        else: 
            f = 9.81*(quad.mq + quad.ml)+2  # thrust force
            tau = 0
        x = quad.runge_kutta_step(quad.x, f, tau)
        X[0:2,i] = x[0:2,0]
        X[2:4,i] = quad.quad_position().flatten()
        Xfull[:,i] = x.flatten()

    # --- animation setup (2D yz plane) ---
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel('y (m)')
    ax.set_ylabel('z (m)')
    # ax.set_title('Quadcopter + Pendulum (yz plane)')
    ax.set_aspect('equal', 'box')

    all_y = np.hstack([X[0, :], X[2, :]])
    all_z = np.hstack([X[1, :], X[3, :]])
    pad = 0.3 + 0.1*quad.l
    ax.set_xlim(all_y.min() - pad, all_y.max() + pad)
    ax.set_ylim(all_z.min() - pad, all_z.max() + pad)

    # trails and markers
    load_trail, = ax.plot([], [], lw=1, label='Load Trajectory')
    quad_trail, = ax.plot([], [], lw=1, label='Quad Trajectory')
    pend_line, = ax.plot([], [], lw=2)
    load_point, = ax.plot([], [], marker='o', markersize=6)
    quad_point, = ax.plot([], [], marker='s', markersize=8)

    # oriented quad: arm line and two motor markers (created once)
    arm_line, = ax.plot([], [], lw=3, solid_capstyle='round', label='Quad arm')
    motor1_point, = ax.plot([], [], marker='o', ms=6)  # +L motor
    motor2_point, = ax.plot([], [], marker='o', ms=6)  # -L motor

    # ax.legend(loc='upper right')

    # If you prefer to keep a list of quad-related artists, define it BEFORE update()
    quad_artists = [arm_line, motor1_point, motor2_point]  # optional, but fine to have

    def init():
        load_trail.set_data([], [])
        quad_trail.set_data([], [])
        pend_line.set_data([], [])
        load_point.set_data([], [])
        quad_point.set_data([], [])
        arm_line.set_data([], [])
        motor1_point.set_data([], [])
        motor2_point.set_data([], [])
        return load_trail, quad_trail, pend_line, load_point, quad_point, arm_line, motor1_point, motor2_point

    def update(i):
        # load position (scalars -> pass as lists)
        y_l, z_l = float(X[0, i]), float(X[1, i])
        # quad position
        y_q, z_q = float(X[2, i]), float(X[3, i])

        # pendulum line (quad -> load)
        pend_line.set_data([y_q, y_l], [z_q, z_l])

        # trails
        load_trail.set_data(X[0, :i+1], X[1, :i+1])
        quad_trail.set_data(X[2, :i+1], X[3, :i+1])

        # points (single-point must be sequences)
        load_point.set_data([y_l], [z_l])
        quad_point.set_data([y_q], [z_q])

        # --- oriented quad arm & motors using phi_q from Xfull ---
        phi_q = float(Xfull[6, i])   # quad angle stored at index 6 in your state
        u_y = np.cos(phi_q)
        u_z = np.sin(phi_q)

        L = quad.L  # distance from center to motor along the body axis
        m1_y = y_q +  L * u_y
        m1_z = z_q +  L * u_z
        m2_y = y_q -  L * u_y
        m2_z = z_q -  L * u_z

        arm_line.set_data([m1_y, m2_y], [m1_z, m2_z])
        motor1_point.set_data([m1_y], [m1_z])
        motor2_point.set_data([m2_y], [m2_z])

        # return ALL artists that changed
        return load_trail, quad_trail, pend_line, load_point, quad_point, arm_line, motor1_point, motor2_point
    gif_folder = "gifs"
    anim = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=50)
    anim.save(f"{gif_folder}/quad_pendulum.gif", writer="pillow", fps=30)

    # In a script use plt.show(); in Jupyter use HTML(anim.to_jshtml())
    plt.show()