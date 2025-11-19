import numpy as np 
from utils import R3_so3
from scipy.linalg import expm
import matplotlib.pyplot as plt

class quad_w_load_dyn:
    def __init__(self, mass_quad=1.0, mass_load=0.5, length=0.3, gravity=9.81):
        self.mq = mass_quad  # mass of the quadcopter
        self.ml = mass_load  # mass of the load
        self.l = length      # length of the cable
        self.g = gravity     # gravitational acceleration
        self.n_states = 15  # number of states in the system
        self.x = np.zeros((self.n_states, 1))  # state vector initialization
        self.R = np.eye(3)  # rotation matrix initialization
        #0:3 position of load
        #3:6 velocity of load
        #6:9 unit vector from load to quadcopter
        #9:12 angular velocity of the load
        #12:15 angular velocity of the quadcopter
        #self.R represents the orientation of the quadcopter
        self.J_quad = np.diag([0.01, 0.01, 0.02])  # inertia matrix of the quadcopter
        self.e_3 = np.array([[0],[0],[1]])  # unit vector in z-direction
        self.dt = 0.01  # time step for integration
        self.h = 0.001 # runge-kutta sub-step size

    # def x_l_dot(self):
    #     """Compute the time derivative of the load position."""
    #     return self.x[3:6]
    
    def v_l_dot(self,p, p_dot, f):
        """Compute the time derivative of the load velocity."""
        aux_1 = p@(f*self.R @ self.e_3)
        aux_2 = p_dot@p_dot
        v_l_dot = (aux_1 - aux_2)*p/(self.ml+self.mq) - self.g*self.e_3
        return v_l_dot
    
    def p_dot(self,p, omega_l):
        """Compute the time derivative of the unit vector from load to quadcopter."""
        p_dot = np.cross(omega_l, p, axis=0)
        print(p_dot.shape)
        input("pause")
        return p_dot
    
    def omega_l_dot(self,p, f):
        """Compute the time derivative of the load's angular velocity."""
        aux = f*self.R @ self.e_3
        omega_l_dot = (-np.cross(p, aux, axis=0))/(self.ml*self.l) 
        return omega_l_dot

    def R_quad_dot(self, omega):
        """Compute the time derivative of the rotation matrix."""
        print(omega.shape)
        input("pause")
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
        # R_dot = self.R_quad_dot()
        x_dot[12:15] = self.omega_quad_dot(omega_quad, tau)
        return x_dot
    def runge_kutta_step(self, x0, f, tau):
        """Perform 4th order integration"""
        n = int(self.dt / self.h)
        for _ in range(n):
            k1_x = self.dynamics(x0, f, tau)
            k2_x = self.dynamics(x0 + 0.5 * self.h * k1_x, f, tau)
            k3_x = self.dynamics(x0 + 0.5 * self.h * k2_x, f, tau)
            k4_x = self.dynamics(x0 + self.h * k3_x, f, tau)

            x0 += (self.h / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            self.R = self.R @ expm(self.h*R3_so3(x0[12:15]))
        self.x = x0

        return x0, self.R
    
    def quad_position(self):
        """Compute the position of the quadcopter."""
        p = self.x[6:9]
        x_l = self.x[0:3]
        x_q = x_l + self.l * p
        return x_q
    
if __name__ == "__main__":
    quad = quad_w_load_dyn()
    f = 9.81*(quad.mq + quad.ml)
    tau = np.array([[0],[0],[0]])
    N = 1000
    X = np.zeros((quad.n_states+3, N))
    x0 = np.zeros((quad.n_states, 1))

    quad.x[6:9] = np.array([[0],[0],[-1]])
    quad.x[3:6] = np.array([[0],[0],[0]])
    quad.x[0:3] = np.array([[0],[0],[0]])

    for i in range(1000):
        quad.x, quad.R = quad.runge_kutta_step(quad.x, f, tau)
        print("Step:", i)
        print("Load Position:", quad.x[0:3].flatten())
        print("Load Velocity:", quad.x[3:6].flatten())
        print("Cable Direction:", quad.x[6:9].flatten())
        print("Load Angular Velocity:", quad.x[9:12].flatten())
        print("Quadcopter Angular Velocity:", quad.x[12:15].flatten())
        print("Rotation Matrix:\n", quad.R)
        print("-------------------------------")