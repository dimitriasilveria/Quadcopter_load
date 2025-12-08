import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def matrix2euler(H):
    """
    Extract Euler angles from homogeneous transformation matrix
    Assumes ZYX rotation order (yaw-pitch-roll)
    """
    R = H[:3, :3]
    
    # Extract angles
    phi = np.arctan2(R[2, 1], R[2, 2])  # roll
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # pitch
    psi = np.arctan2(R[1, 0], R[0, 0])  # yaw
    
    return psi, phi, theta


class QuadLoad3dSimple:
    """
    QUADCOPTER
    THIS IS THE 3D VERSION OF A QUADCOPTER PLOT CLASS
    """
    
    def __init__(self, q, arm_length, height, cable_length, samples, h_quad):
        """
        Initialize the quadrotor with load visualization
        
        Parameters:
        -----------
        q : array
            Initial state vector [24x1]
        arm_length : float
            Length of quadrotor arm
        height : float
            Height of quadrotor
        cable_length : float
            Length of cable to load
        samples : int
            Number of trajectory samples
        h_quad : matplotlib axes
            Axes handle for plotting
        """
        self.arm_length = arm_length
        self.height = height
        self.cable_length = cable_length
        
        self.samples = samples + 1
        
        # State variables
        self.q = np.zeros(24)
        self.dq = np.zeros(24)
        self.p = np.zeros(3)
        
        # Trajectory storage
        self.q_traj = np.zeros((24, samples))
        self.dq_traj = np.zeros((24, samples))
        self.pose_q_traj = np.zeros((4, samples))
        self.pos_l_traj = np.zeros((3, samples))
        self.p_traj = np.zeros((3, samples))
        self.psi_q_traj = np.zeros(samples)
        self.psi_l_traj = np.zeros(samples)
        self.pose_des_traj = np.zeros((4, samples))
        self.vel_des_traj = np.zeros((4, samples))
        self.acc_des_traj = np.zeros((3, samples))
        self.t_traj = np.zeros(samples)
        
        self.f_traj = np.zeros(samples)
        self.m_traj = np.zeros((3, samples))
        
        # Current state
        self.pos_l = None
        self.pos_q = None
        self.rot = None
        self.Hb2w = None
        self.t = None
        self.plot_pos = None
        self.pos_q_plot = None
        self.vector_plot = None
        
        self.counter = 0
        
        # Update initial state
        self.update_state(q, self.dq)
        
        # Initialize plot handles
        self.h_q = h_quad
        self.h_q.hold = True
        
        self.h_pose_traj = self.h_q.plot([], [], [], 'b', linewidth=2)[0]
        self.h_des_traj = self.h_q.plot([], [], [], 'r.', linewidth=1.5)[0]
        
        self.h_qm13 = self.h_q.plot(
            self.plot_pos[0, [0, 2]], 
            self.plot_pos[1, [0, 2]], 
            self.plot_pos[2, [0, 2]], 
            '-ko', markerfacecolor='blue', markersize=5
        )[0]
        
        self.h_qm24 = self.h_q.plot(
            self.plot_pos[0, [1, 3]], 
            self.plot_pos[1, [1, 3]], 
            self.plot_pos[2, [1, 3]], 
            '-ko', markerfacecolor='blue', markersize=5
        )[0]
        
        self.h_q_norm = self.h_q.plot(
            self.plot_pos[0, [4, 5]], 
            self.plot_pos[1, [4, 5]], 
            self.plot_pos[2, [4, 5]], 
            '-ko', markerfacecolor='red', markersize=5
        )[0]
        
        self.h_load = self.h_q.plot(
            [self.pos_q[0], self.pos_l[0]], 
            [self.pos_q[1], self.pos_l[1]], 
            [self.pos_q[2], self.pos_l[2]], 
            '-ro', markerfacecolor='red', markersize=5
        )[0]
        
        self.h_quiver = self.h_q.quiver(
            [], [], [], [], [], [], 
            linewidth=1.5
        )
    
    def step(self, t, q, dq, F, M, des_traj, psi_q, psi_l, vector_plot):
        """
        Update and plot one simulation step
        """
        self.update_state(q, dq)
        self.update_traj(t, des_traj, F, M, psi_q, psi_l)
        
        self.vector_plot = vector_plot
        self.plot_quad()
    
    def update_state(self, q, dq):
        """
        Update current state from state vector
        """
        self.q = q
        self.dq = dq
        self.pos_l = np.array([self.q[0], self.q[1], self.q[2]])
        self.p = self.q[6:9]
        self.pos_q = self.pos_l - self.cable_length * self.p
        
        self.rot = self.q[12:21].reshape(3, 3, order='F')
        
        self.Hb2w = np.vstack([
            np.column_stack([self.rot, self.pos_q]),
            [0, 0, 0, 1]
        ])
        
        self.plot_pos = self.quad_positions()
        self.pos_q_plot = np.tile(self.pos_q.reshape(-1, 1), (1, 5))
    
    def update_traj(self, t, des_traj, F, M, psi_q, psi_l):
        """
        Update trajectory data
        """
        self.t_traj[self.counter] = t
        self.q_traj[:, self.counter] = self.q
        self.dq_traj[:, self.counter] = self.dq
        self.pos_l_traj[:, self.counter] = self.pos_l
        self.p_traj[:, self.counter] = self.p
        self.pose_q_traj[0:3, self.counter] = self.pos_q
        
        psi, phi, theta = matrix2euler(self.Hb2w)