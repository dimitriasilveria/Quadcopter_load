from quad_w_load_dyn import quad_w_load_dyn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_controller import load_controller
from quad_controller import quad_controller
from utils import animate_quad_and_load, skew_to_R3
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
    N = 50
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
    Rc_state  = np.vstack((np.eye(3),np.zeros((3,3))))  # desired rotation and its derivative for filtering
    pc_state = np.zeros((6,1))  # desired load position and its derivative for filtering
    pc_state[2,0] = -1.0  # initial desired load position
    #storage for plotting
    x_load = np.zeros((3,N))
    x_quad = np.zeros((3,N))
    R_load = np.zeros((3,3,N))
    for i in range(N):
        #load position controller
        p_des, F_n = load_ctrl.position_controller(x_des[:,i].reshape((3,1)), v_des[:,i].reshape((3,1)), a_des[:,i].reshape((3,1)))
        #load attitude controller
        pc, dpc, ddpc = load_ctrl.command_filter(pc_state, p_des, 0.98, 7)
        p_dot_des = dpc
        p_ddot_des = ddpc
        pc_state = np.vstack((pc, dpc))
        R_des, F = load_ctrl.attitude_controller(p_des, p_dot_des, p_ddot_des, F_n, b1_d)
        Rc, dRc, ddRc = load_ctrl.command_filter(Rc_state, R_des, 0.98, 75)
        Rc = Rc/np.linalg.det(Rc)**(1/3)
        Rc_state = np.vstack((Rc, dRc))
        dR_des = dRc
        w_des = skew_to_R3(R_des.T @ dR_des)
        
        #quadcopter controller
        tau, f = quad_ctrl.controller(R_des, w_des, F)
        #dynamics update
        quad.x, quad.R = quad.runge_kutta_step(quad.x, f, tau)
        #storing for plotting
        x_load[:,i] = quad.x[0:3].flatten()
        x_quad[:,i] = quad.quad_position().flatten()
        R_load[:,:,i] = quad.R

    fig, ax, anim = animate_quad_and_load(x_load, x_quad, R=R_load, x_des=x_des, trail=60)
    anim.save('quad_with_load_controller.gif', writer='pillow', fps=30)