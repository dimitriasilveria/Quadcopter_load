from quad_w_load_dyn import quad_w_load_dyn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_controller import load_controller
from quad_controller import quad_controller
from utils import animate_quadcopter_pendulum_3d, animate_quadcopter_pendulum_with_desired

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
    load_ctrl = load_controller(quad)
    quad_ctrl = quad_controller(quad)
    #storage for plotting
    x_load = np.zeros((3,N))
    x_quad = np.zeros((3,N))
    R_load = np.zeros((3,3,N))
    for i in range(N):
        #load position controller
        p_des, F_n = load_ctrl.position_controller(x_des[:,i].reshape((3,1)), v_des[:,i].reshape((3,1)), a_des[:,i].reshape((3,1)))
        #load attitude controller
        R_des, F = load_ctrl.attitude_controller(p_des, np.zeros((3,1)), np.zeros((3,1)), F_n, b1_d)
        #quadcopter controller
        tau, f = quad_ctrl.controller(R_des, np.zeros((3,1)), F)
        #dynamics update
        x_dot = quad.dynamics(quad.x, f, tau)
        quad.x, quad.R = quad.runge_kutta_step(quad.x, f, tau)
        #storing for plotting
        x_load[:,i] = quad.x[0:3].flatten()
        x_quad[:,i] = quad.quad_position().flatten()
        R_load[:,:,i] = quad.R


    fig, ax, anim = animate_quadcopter_pendulum_with_desired(np.vstack((x_load, x_quad)), Rot=R_load, X_des=x_des, cable_length=quad.l)
    anim.save("quadcopter_trajectory.gif", writer="pillow", fps=30)