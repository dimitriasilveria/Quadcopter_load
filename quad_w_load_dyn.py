import numpy as np 
from utils import R3_so3
from scipy.linalg import expm
import matplotlib.pyplot as plt
from icecream import ic
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from utils import animate_quadcopter_pendulum_3d

class quad_w_load_dyn:
    def __init__(self, mass_quad=0.835, mass_load=0.088, length=0.5, gravity=9.81):
        self.mq = mass_quad  # mass of the quadcopter
        self.ml = mass_load  # mass of the load
        self.l = length      # length of the cable
        self.g = gravity     # gravitational acceleration
        self.n_states = 15  # number of states in the system
        self.x = np.zeros((self.n_states, 1))  # state vector initialization
        self.x[6:9] = np.array([[0],[0],[-1]])  # initial unit vector from load to quadcopter
        self.R = np.eye(3)  # rotation matrix initialization
        #0:3 position of load
        #3:6 velocity of load
        #6:9 unit vector from load to quadcopter
        #9:12 angular velocity of the load
        #12:15 angular velocity of the quadcopter
        #self.R represents the orientation of the quadcopter
        self.J_quad = 1e-3*np.diag([2.32, 2.32, 4])  # inertia matrix of the quadcopter
        self.J_quad_inv = np.linalg.inv(self.J_quad)
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
        omega_quad_dot = self.J_quad_inv @ (tau - np.cross(omega, self.J_quad @ omega, axis=0))
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
        x_q = x_l - self.l * p
        return x_q
    

if __name__ == "__main__":
    quad = quad_w_load_dyn()
    f = 9.81*(quad.mq + quad.ml)+2  # thrust force
    tau = np.array([[0.],[0.01],[0]])
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
            f = 9.81*(quad.mq + quad.ml)+2  # thrust force
            # tau = np.array([[0],[-0.01],[0]])
        else: 
            f = 9.81*(quad.mq + quad.ml)  # thrust force
            tau = np.array([[0],[0.0],[0]])
        x, R_ = quad.runge_kutta_step(quad.x, f, tau)
        X[0:3,i] = x[0:3,0]
        X[3:6,i] = quad.quad_position().flatten()
        Rot[:,:,i] = R_

    # plot_quadcopter_pendulum_3d(X, Rot=Rot, cable_length=quad.l, indices=[0, N//2, N-1])
    fig, ax, anim = animate_quadcopter_pendulum_3d(X, Rot=Rot, cable_length=quad.l,
                                              quad_arm_length=0.12, interval=40, trail=60)
    anim.save("quadcopter.gif", writer="pillow", fps=30)
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