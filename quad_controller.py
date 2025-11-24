import numpy as np
from utils import R3_so3,skew_to_R3
from icecream import ic
#0:3 position of load
#3:6 velocity of load
#6:9 unit vector from load to quadcopter
#9:12 angular velocity of the load
#12:15 angular velocity of the quadcopter
#self.R represents the orientation of the quadcopter
class quad_controller:
    def __init__(self, quad_dyn):
        self.quad_dyn = quad_dyn
        self.g = quad_dyn.g
        self.mq = quad_dyn.mq
        self.ml = quad_dyn.ml
        self.l = quad_dyn.l
        self.J_quad = quad_dyn.J_quad
        self.k_R = np.diag([10.4,15.4,0.4])
        self.k_omega = np.diag([0.12,0.12,0.08])

    def controller(self, R_des, omega_des, F):
        Rot = self.quad_dyn.R
        omega = self.quad_dyn.x[12:15]
        aux =  R3_so3(omega) @ Rot.T @ R_des @ omega_des - Rot.T @ R_des @ omega_des
        e_R = 0.5 * skew_to_R3(R_des.T @ Rot - Rot.T @ R_des)
        e_R = e_R.reshape((3,1))
        e_omega = omega - Rot.T @ R_des @ omega_des
        tau = - self.k_R @ e_R - self.k_omega @ e_omega + np.cross(omega, self.J_quad @ omega, axis=0) - self.J_quad @ aux
        f = F.T @ Rot@self.quad_dyn.e_3
        return tau, f