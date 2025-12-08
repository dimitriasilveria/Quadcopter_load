import numpy as np
from quad_w_load_dyn_2D import quad_w_load_dyn

class Controller_2D:
    def __init__(self, quad_dyn):
        self.quad_dyn = quad_dyn
        self.g = quad_dyn.g
        self.mq = quad_dyn.mq
        self.ml = quad_dyn.ml
        self.J = quad_dyn.J_quad
        self.l = quad_dyn.l
        self.kx = np.diag([10.4,15.4])
        self.kv = np.diag([6.0,9.5])
        self.k_p_l = 9.0
        self.k_d_l = 2.5
        self.k_p = 8.0
        self.k_d = 2.0
        self.e_3 = quad_dyn.e_3

    def load_position_controller(self, x_des, v_des, a_des, p_ddot_des):
        e_x = self.quad_dyn.x[0:2] - x_des
        e_v = self.quad_dyn.x[2:4] - v_des
        A = -self.kx @ e_x - self.kv @ e_v + (self.ml) * (a_des + self.g * self.e_3)
        B = self.mq*a_des - self.mq*self.l*p_ddot_des + self.mq*self.g*self.e_3
        Rot = self.quad_dyn.R
        f = (A + B).T@Rot@self.e_3
        phi_l_des = np.arctan2(-A[0], A[1])
        return phi_l_des, f
    
    def load_attitude_controller(self, phi_l_des, w_l_des, w_l_dot_des, f):
        phi_l = self.quad_dyn.x[4,0]
        w_l = self.quad_dyn.x[5,0]
        e_l = phi_l - phi_l_des
        e_l_dot = w_l - w_l_des
        aux = -self.k_p*e_l - self.k_d*e_l_dot + (self.mq*self.l*w_l_dot_des)/(f)
        print(aux)
        input()
        aux = np.clip(aux, -1, 1)
        phi_q_des = phi_l + np.arcsin(aux)
        return phi_q_des
    
    def quad_attitude_controller(self, phi_q_des, w_q_des,w_q_dot_des):
        phi_q = self.quad_dyn.x[6,0]
        w_q = self.quad_dyn.x[7,0]
        e_q = phi_q - phi_q_des
        e_q_dot = w_q - w_q_des
        tau = self.J[0,0]*(- self.k_p * e_q - self.k_d * e_q_dot + w_q_dot_des)
        return tau
    
if __name__ == "__main__":
    quad = quad_w_load_dyn()
    #reference trajectory for the load as a circle
    N = 300
    r = 1.0
    w = 2*np.pi/10
    t = np.linspace(0, N*quad.dt, N)
    x_l_des = np.array([r*np.sin(w*t),r*np.cos(w*t)])
    v_l_des = np.array([w*r*np.cos(w*t),-w*r*np.sin(w*t)])
    a_l_des = np.array([-w**2*r*np.sin(w*t),-w**2*r*np.cos(w*t)])
    #P_DDOT_DES IS NOT A_DES

    controller = Controller_2D(quad)
    quad.x[0:2] = x_l_des[:,0].reshape((2,1))
    quad.x[2:4] = v_l_des[:,0].reshape((2,1))
    phi_q = 0.0
    quad.R = np.array([[np.cos(phi_q), -np.sin(phi_q)],
                       [np.sin(phi_q),  np.cos(phi_q)]])
    #storage for plotting
    x_load = np.zeros((2,N))
    x_quad = np.zeros((2,N))
    for i in range(N):
        #load position controller
        p_ddot_des = a_l_des[:,i].reshape((2,1))
        phi_l_des, f = controller.load_position_controller(x_l_des[:,i].reshape((2,1)), v_l_des[:,i].reshape((2,1)), a_l_des[:,i].reshape((2,1)), p_ddot_des)
        #load attitude controller
        w_l_des = 0.0
        w_l_dot_des = 0.0
        phi_q_des = controller.load_attitude_controller(phi_l_des, w_l_des, w_l_dot_des, f)
        #quadcopter attitude controller
        w_q_des = 0.0
        w_q_dot_des = 0.0
        tau = controller.quad_attitude_controller(phi_q_des, w_q_des, w_q_dot_des)
        #dynamics update
        quad.x = quad.runge_kutta_step(quad.x, f, tau)
        #storing for plotting
        x_load[:,i] = quad.x[0:2].flatten()
        x_quad[:,i] = quad.quad_position().flatten()