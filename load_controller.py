import numpy as np

class load_controller:
    def __init__(self, quad_dyn):
        self.quad_dyn = quad_dyn
        self.g = quad_dyn.g
        self.mq = quad_dyn.mq
        self.ml = quad_dyn.ml
        self.l = quad_dyn.l
        self.g = quad_dyn.g
        self.kx = 8.0
        self.kv = 6.0
        self.kp = 10.0
        self.kw = 4.0
        self.e_3 = quad_dyn.e_3

        

    def position_controller(self, x_des, v_des, a_des):
        e_x = self.quad_dyn.x[0:3] - x_des
        e_v = self.quad_dyn.x[3:6] - v_des
        p = self.quad_dyn.x[6:9]
        p_dot = self.quad_dyn.p_dot(p, self.quad_dyn.x[9:12])

        A = -self.kx * e_x - self.kv * e_v + (self.ml + self.mq) * (a_des + self.g * self.e_3)+ self.mq*self.l *(p_dot.T @ p_dot) * p
        p_c = -A / np.linalg.norm(A)
        F_n = (A.T @ p)*p
        return p_c, F_n
    
    def attitude_controller(self,p_des, p_dot_des, p_ddot_des, F_n, b1_d):
        p = self.quad_dyn.x[6:9]
        p_dot = self.quad_dyn.p_dot(p, self.quad_dyn.x[9:12])
        e_p = p - p_des
        e_p_dot = p_dot - p_dot_des
        F_pd = -self.kp * e_p - self.kw * e_p_dot
        aux = self.mq*self.l*(np.cross(p_des, p_ddot_des, axis=0))
        F_ff = self.ml * self.l * (p.T @ (p_dot_des.T@ p_dot_des)) * (np.cross(p,p_dot_des, axis=0)) + np.cross(aux, p, axis=0)
        F = F_n - F_pd - F_ff
        b3_c = -F / np.linalg.norm(F)
        b1_c = - np.cross(b3_c, np.cross(b3_c, b1_d, axis=0), axis=0)/ np.linalg.norm(np.cross(b3_c, b1_d, axis=0))
        b2_c = np.cross(b3_c, b1_c, axis=0)
        R_c = np.hstack((b1_c, b2_c, b3_c))
        return R_c, F