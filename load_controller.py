import numpy as np
from icecream import ic
#0:3 position of load
#3:6 velocity of load
#6:9 unit vector from load to quadcopter
#9:12 angular velocity of the load
#12:15 angular velocity of the quadcopter
#self.R represents the orientation of the quadcopter
class load_controller:
    def __init__(self, quad_dyn):
        self.quad_dyn = quad_dyn
        self.g = quad_dyn.g
        self.mq = quad_dyn.mq
        self.ml = quad_dyn.ml
        self.l = quad_dyn.l
        self.g = quad_dyn.g
        self.kx = np.diag([5.4,5.4,5.85])
        self.kv = np.diag([6.0,6.0,4.5])
        self.kp = 9
        self.kw = 7.5
        self.e_3 = quad_dyn.e_3


        

    def position_controller(self, x_des, v_des, a_des):
        e_x = self.quad_dyn.x[0:3] - x_des
        e_v = self.quad_dyn.x[3:6] - v_des
        p = self.quad_dyn.x[6:9]
        p_dot = self.quad_dyn.p_dot(p, self.quad_dyn.x[9:12])
        A = -self.kx @ e_x - self.kv @ e_v + (self.ml + self.mq) * (a_des + self.g * self.e_3)+ self.mq*self.l *(p_dot.T @ p_dot) * p
        p_c = -A / np.linalg.norm(A)
        F_n = (p_c.T @ p)*p
        return p_c, F_n
    
    def attitude_controller(self,p_des, p_dot_des, p_ddot_des, F_n, b1_d):
        p = self.quad_dyn.x[6:9]
        p_dot = self.quad_dyn.p_dot(p, self.quad_dyn.x[9:12])
        e_p = p - p_des
        e_p_dot = p_dot - p_dot_des
        F_pd = -self.kp * e_p - self.kw * e_p_dot
        aux = self.mq*self.l*(np.cross(p_des, p_ddot_des, axis=0))
        F_ff = self.ml * self.l * (p.T @ (np.cross(p_des, p_dot_des, axis=0))) * (np.cross(p,p_dot_des, axis=0)) + np.cross(aux, p, axis=0)
        F = F_n - F_pd - F_ff
        b3_c = -F / np.linalg.norm(F)
        b1_c = - np.cross(b3_c, np.cross(b3_c, b1_d, axis=0), axis=0)/ np.linalg.norm(np.cross(b3_c, b1_d, axis=0))
        b2_c = np.cross(b3_c, b1_c, axis=0)
        R_c = np.hstack((b1_c, b2_c, b3_c))
        return R_c, F
    
    def second_order_filter(self, x, xc, zeta=0.7, wn=30):
        if x.shape[1] == 3:
            dx2 = -2*zeta*wn*x[3:6,:] + wn**2 * (x[0:3,:] - xc)
            # dx2 = dx2/np.linalg.det(dx2)
            dx = np.vstack((x[3:6,:], dx2))
        else:
            dx2 = -2*zeta*wn*x[3:6] + wn**2 * (x[0:3] - xc)
            dx = np.vstack((x[3:6], dx2))
        return dx
    
    def runge_kutta_step(self, x, xc, zeta=0.7, wn=30):
        h = self.quad_dyn.dt
        k1 = self.second_order_filter(x, xc, zeta, wn)
        k2 = self.second_order_filter(x + 0.5*h*k1, xc, zeta, wn)
        k3 = self.second_order_filter(x + 0.5*h*k2, xc, zeta, wn)
        k4 = self.second_order_filter(x + h*k3, xc, zeta, wn)
        x_next = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        return x_next
    
    def command_filter(self, x, xc, zeta=0.7, wn=30):
        x_next = self.runge_kutta_step(x, xc, zeta, wn)
        d_state = self.second_order_filter(x_next, xc, zeta, wn)
        x = x_next[0:3]
        dx = x_next[3:6]
        ddx = d_state[3:6]
        return x, dx, ddx