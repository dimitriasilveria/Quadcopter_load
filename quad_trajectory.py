from quad_w_load_dyn import quad_w_load_dyn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_controller import load_controller
from quad_controller import quad_controller
from utils import animate_quad_and_load, calc_w_from_Rdot
from icecream import ic
from scipy.spatial.transform import Rotation as R
#0:3 position of load
#3:6 velocity of load
#6:9 unit vector from load to quadcopter
#9:12 angular velocity of the load
#12:15 angular velocity of the quadcopter
#self.R represents the orientation of the quadcopter
if __name__ == "__main__":
    #defining reference trajectories
    N = 500
    t = np.linspace(0, N*0.01, N)
    x_des = np.array([0.5*np.sin(0.1*t),0.5*np.cos(0.1*t),-0.5*np.ones(N)])
    v_des = np.array([0.05*np.cos(0.1*t),-0.05*np.sin(0.1*t),np.zeros(N)])
    a_des = np.array([-0.005*np.sin(0.1*t),-0.005*np.cos(0.1*t),np.zeros(N)])
    b1_d = np.array([[1],[0],[0]])
    #initializing dynamics and controllers
    quad = quad_w_load_dyn()
    quad.x[0:3] = x_des[:,0].reshape((3,1))
    quad.x[3:6] = v_des[:,0].reshape((3,1))
    load_ctrl = load_controller(quad)
    quad_ctrl = quad_controller(quad)
    w_des = np.zeros((3,1))
    R_prev_des = np.eye(3)
    R_des = np.eye(3)
    #storage for plotting
    x_load = np.zeros((3,N))
    x_quad = np.zeros((3,N))
    R_load = np.zeros((3,3,N))
    for i in range(N):
        #load position controller
        p_des, F_n = load_ctrl.position_controller(x_des[:,i].reshape((3,1)), v_des[:,i].reshape((3,1)), a_des[:,i].reshape((3,1)))
        #load attitude controller
        R_prev_des = R_des
        R_des, F = load_ctrl.attitude_controller(p_des, np.zeros((3,1)), np.zeros((3,1)), F_n, b1_d)
        if i > 0:
            ic(R.from_matrix(R_des).as_euler('xyz', degrees=True), R.from_matrix(R_prev_des).as_euler('xyz', degrees=True))
            ic(R.from_matrix(R_prev_des.T @ R_des).as_euler('xyz', degrees=True))
            input("wait")
            w_des = calc_w_from_Rdot(R_des, R_prev_des, quad.dt)
        #quadcopter controller
        tau, f = quad_ctrl.controller(R_des, w_des, F)
        #dynamics update
        x_dot = quad.dynamics(quad.x, f, tau)
        quad.x, quad.R = quad.runge_kutta_step(quad.x, f, tau)
        #storing for plotting
        x_load[:,i] = quad.x[0:3].flatten()
        x_quad[:,i] = quad.quad_position().flatten()
        R_load[:,:,i] = quad.R

    fig, ax, anim = animate_quad_and_load(x_load, x_quad, R=R_load, x_des=x_des, trail=60)
    anim.save('quad_with_load_controller.gif', writer='pillow', fps=30)