import numpy as np 

class quad_w_load_dyn:
    def __init__(self, mass_quad=1.0, mass_load=0.5, length=0.3, gravity=9.81):
        self.mq = mass_quad  # mass of the quadcopter
        self.ml = mass_load  # mass of the load
        self.l = length      # length of the cable
        self.g = gravity     # gravitational acceleration
        self.n_states = 19  # number of states in the system
        self.x = np.zeros((self.n_states, 1))  # state vector initialization
        #0:3 position of load
        #3:6 velocity of load
        #6:9 unit vector from load to quadcopter
        #9:12 angular velocity of the load
        #12:16 orientation quaternion of the quadcopter
        #16:19 angular velocity of the quadcopter
        self.J_quad = np.diag([0.01, 0.01, 0.02])  # inertia matrix of the quadcopter

    def x_l_dot(self):
        """Compute the time derivative of the load position."""
        return self.x[3:6]
    
    def q_dot(self):
        """Compute the time derivative of the orientation quaternion."""
        omega = self.x[16:19]
        q = self.x[12:16].flatten()
        Omega = np.array([[0, -omega[0], -omega[1], -omega[2]],
                          [omega[0], 0, omega[2], -omega[1]],
                          [omega[1], -omega[2], 0, omega[0]],
                          [omega[2], omega[1], -omega[0], 0]])
        q_dot = 0.5 * Omega @ q
        return q_dot.reshape((4, 1))
