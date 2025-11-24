import numpy as np 
from utils import R2_so2, animate_quad_and_load
from scipy.linalg import expm
import matplotlib.pyplot as plt
from icecream import ic
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
#import ode from scipy
from scipy.integrate import RK45

class quad_w_load_dyn:
    def __init__(self, mass_quad=0.835, mass_load=0.088, length=0.5, gravity=9.81):
        self.mq = mass_quad  # mass of the quadcopter
        self.ml = mass_load  # mass of the load
        self.l = length      # length of the cable
        self.g = gravity     # gravitational acceleration
        self.n_states = 10  # number of states in the system
        self.x = np.zeros((self.n_states, 1))  # state vector initialization
        self.x[4:6] = np.array([[0],[-1]])  # initial unit vector from load to quadcopter
        self.R = np.eye(2)  # rotation matrix initialization
        #0:2 position of load
        #2:4 velocity of load
        #4:6 unit vector from load to quadcopter
        #6:8 angular velocity of the load
        #8:10 angular velocity of the quadcopter
        #self.R represents the orientation of the quadcopter
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
        # tau_x_max = self.L * self.kf * (self.w_max**2 - self.w_min**2)
        tau_y_max = self.L * self.kf * (self.w_max**2 - self.w_min**2)
        tau_z_max = 2*self.km * (self.w_max**2 - self.w_min**2)
        max_tau = np.array([[tau_y_max],[tau_z_max]])
        max_thrust = self.kf * 4 * self.w_max**2
        return max_thrust, max_tau
    
    def calc_min_torque_thrust(self):
        # tau_x_min = self.L * self.kf * (self.w_min**2 - self.w_max**2)
        tau_y_min = self.L * self.kf * (self.w_min**2 - self.w_max**2)
        tau_z_min = 2*self.km * (self.w_min**2 - self.w_max**2)
        min_tau = np.array([[tau_y_min],[tau_z_min]])
        min_thrust = self.kf * 4 * self.w_min**2
        return min_thrust, min_tau
    
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
        omega_hat = R2_so2(omega)
        R_dot = omega_hat @ self.R
        return R_dot
    
    def omega_quad_dot(self, omega, tau):
        """Compute the time derivative of the quadcopter's angular velocity."""
        omega_quad_dot = self.J_quad_inv @ (tau - np.cross(omega, self.J_quad @ omega, axis=0))
        return omega_quad_dot
    
    #0:2 position of load
    #2:4 velocity of load
    #4:6 unit vector from load to quadcopter
    #6:8 angular velocity of the load
    #8:10 angular velocity of the quadcopter
    #self.R represents the orientation of the quadcopter
    def dynamics(self,x, f, tau):
        """Compute the time derivative of the full state."""
        omega_l = x[6:8]
        omega_quad = x[8:10]
        p = x[4:6]
        v_l = x[2:4]
        p_dot = self.p_dot(p, omega_l)
        x_dot = np.zeros((self.n_states, 1))
        x_dot[0:2] = v_l
        x_dot[2:4] = self.v_l_dot(p, p_dot, f)
        x_dot[4:6] = p_dot
        x_dot[6:8] = self.omega_l_dot(p, f)
        # R_dot = self.R_quad_dot(omega_quad)
        x_dot[8:10] = self.omega_quad_dot(omega_quad, tau)
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
        self.R = self.R @ expm(self.dt*R2_so2(x0[8:10]))
        #limiting angular velocity of quadcopter
        x0[8:10] = np.clip(x0[8:10], -np.pi, np.pi)
        self.x = x0
        return x0, self.R
    
    def quad_position(self):
        """Compute the position of the quadcopter."""
        p = self.x[4:6]
        x_l = self.x[0:2]
        x_q = x_l - self.l * p
        return x_q
    

if __name__ == "__main__":
    quad = quad_w_load_dyn()
    f = 9.81*(quad.mq + quad.ml)+2  # thrust force
    tau = np.array([[0.01],[0]])
    N = 200
    t = np.linspace(0, N*quad.dt, N)
    Tau = np.array([np.zeros(N), np.sin(0.01*t), np.zeros(N)])
    f = 9.81*(quad.mq + quad.ml)  # thrust force
    Rot = np.zeros((2,2,N))
    X = np.zeros((4, N))

    x0 = np.zeros((quad.n_states, 1))
    X[0:2,0] = x0[0:2,0]
    
    # quad.x[6:9] = x0[6:9] = np.array([[1],[1],[1]])/np.linalg.norm(np.array([[1],[1],[1]]))
    X[2:4,0] = quad.quad_position().flatten()

    for i in range(N):
        if i == 0:
            f = 9.81*(quad.mq + quad.ml)+2  # thrust force
            # tau = np.array([[0],[-0.01],[0]])
        else: 
            f = 9.81*(quad.mq + quad.ml)  # thrust force
            tau = np.array([[0],[0.0]])
        x, R_ = quad.runge_kutta_step(quad.x, f, tau)
        X[0:2,i] = x[0:2,0]
        X[2:4,i] = quad.quad_position().flatten()
        Rot[:,:,i] = R_

    # plot_quadcopter_pendulum_3d(X, Rot=Rot, cable_length=quad.l, indices=[0, N//2, N-1])
    # fig, ax, anim = animate_quad_and_load(X[0:2,:], X[2:4,:], R=Rot, x_des=None,
    #                                           quad_arm_length=0.12, interval=40, trail=60)
    # anim.save("quadcopter.gif", writer="pillow", fps=30)

    #plot 2D trajectories

    fig = plt.figure()
    plt.plot(X[0,:], X[1,:], label='Load Trajectory')
    plt.plot(X[2,:], X[3,:], label='Quadcopter Trajectory')
    # quad.draw_pendulum_3d(ax, X[3:6,-1], X[0:3,-1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # quad.animate_quad_pendulum_3d(X[3:6,:], X[0:3,:], Rot=Rot, interval=50, elev=20, azim=-60)