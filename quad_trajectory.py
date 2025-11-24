from quad_w_load_dyn import quad_w_load_dyn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from load_controller import load_controller
from quad_controller import quad_controller
from utils import animate_quad_and_load, skew_to_R3
from icecream import ic
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
#0:3 position of load
#3:6 velocity of load
#6:9 unit vector from load to quadcopter
#9:12 angular velocity of the load
#12:15 angular velocity of the quadcopter
#self.R represents the orientation of the quadcopter
if __name__ == "__main__":
    #defining reference trajectories
    N = 120
    quad = quad_w_load_dyn()
    r = 1.0
    t = np.linspace(0, N*quad.dt, N)
    x_des = np.array([r*np.sin(0.1*t),r*np.cos(0.1*t),0.5*np.ones(N)])
    v_des = np.array([0.1*r*np.cos(0.1*t),-0.1*r*np.sin(0.1*t),np.zeros(N)])
    a_des = np.array([-0.01*r*np.sin(0.1*t),-0.01*r*np.cos(0.1*t),np.zeros(N)])
    W_des = np.zeros_like(x_des)
    w_quad = np.zeros_like(x_des)
    b1_d = np.array([[1],[0],[0]])
    #initializing dynamics and controllers
    
    quad.x[0:3] = x_des[:,0].reshape((3,1))
    quad.x[3:6] = v_des[:,0].reshape((3,1))
    load_ctrl = load_controller(quad)
    quad_ctrl = quad_controller(quad)
    R_des_prev = quad.R
    Rc_state  = np.vstack((np.eye(3),np.zeros((3,3))))  # desired rotation and its derivative for filtering
    pc_state = np.zeros((6,1))  # desired load position and its derivative for filtering
    pc_state[2,0] = -1.0  # initial desired load position
    #storage for plotting
    x_load = np.zeros((3,N))
    x_quad = np.zeros((3,N))
    R_load = np.zeros((3,3,N))
    v_load = np.zeros((3,N))
    for i in range(N):
        #load position controller
        p_des, A = load_ctrl.position_controller(x_des[:,i].reshape((3,1)), v_des[:,i].reshape((3,1)), a_des[:,i].reshape((3,1)))
        #load attitude controller
        pc, dpc, ddpc = load_ctrl.command_filter(pc_state, p_des, 0.98, 7)
        p_dot_des = dpc
        p_ddot_des = ddpc
        pc_state = np.vstack((pc, dpc))
        R_des, F = load_ctrl.attitude_controller(p_des, p_dot_des, p_ddot_des, A, b1_d)
        # Rc, dRc, ddRc = load_ctrl.command_filter(Rc_state, R_des, 0.98, 75)
        # Rc = Rc/np.linalg.det(Rc)**(1/3)
        # input()
        # Rc_state = np.vstack((Rc, dRc))
        # dR_des = dRc
        w_des = (1/quad.dt)*skew_to_R3(logm(R_des_prev.T @ R_des))
        W_des[:,i] = w_des.flatten()

        # w_des = skew_to_R3(R_des.T @ dR_des)
        # input()        
        #quadcopter controller
        tau, f = quad_ctrl.controller(R_des, w_des, F)
        #dynamics update
        quad.x, quad.R = quad.runge_kutta_step(quad.x, f, tau)
        #storing for plotting
        x_load[:,i] = quad.x[0:3].flatten()
        x_quad[:,i] = quad.quad_position().flatten()
        w_quad[:,i] = quad.x[12:15].flatten()
        v_load[:,i] = quad.x[3:6].flatten()
        R_des_prev = R_des
        # quad.x[0:3] = x_des[:,i].reshape((3,1))  # position reset for trajectory tracking
        # quad.x[3:6] = v_des[:,i].reshape((3,1))  # velocity reset for trajectory tracking
        R_load[:,:,i] = quad.R

    # fig, ax, anim = animate_quad_and_load(x_load, x_quad, R=R_load, x_des=x_des, trail=60, interval = 5)
    # anim.save('quad_with_load_controller.gif', writer='pillow', fps=60)

    #plotting load desired and actual trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x_des[0,:], x_des[1,:], x_des[2,:], label='Desired Load Trajectory')
    ax.plot(x_load[0,:], x_load[1,:], x_load[2,:], label='Actual Load Trajectory')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    # plt.show()

    #plotting load desired and actual trajectory in time
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(t, x_des[0,:], label='Desired Load X')
    axs[0].plot(t, x_load[0,:], label='Actual Load X')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('X Position (m)')
    axs[0].legend()
    axs[1].plot(t, x_des[1,:], label='Desired Load Y')
    axs[1].plot(t, x_load[1,:], label='Actual Load Y')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Y Position (m)')
    axs[1].legend()
    axs[2].plot(t, x_des[2,:], label='Desired Load Z')
    axs[2].plot(t, x_load[2,:], label='Actual Load Z')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z Position (m)')
    axs[2].legend()
    plt.show()

    #plotting desired and actual load velocity in time
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(t, v_des[0,:], label='Desired Load Vx')
    axs[0].plot(t, v_load[0,:], label='Actual Load Vx')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('X Velocity (m/s)')
    axs[0].legend()
    axs[1].plot(t, v_des[1,:], label='Desired Load Vy')
    axs[1].plot(t, v_load[1,:], label='Actual Load Vy')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Y Velocity (m/s)')
    axs[1].legend()
    axs[2].plot(t, v_des[2,:], label='Desired Load Vz')
    axs[2].plot(t, v_load[2,:], label='Actual Load Vz')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Z Velocity (m/s)')
    axs[2].legend()
    # plt.show()

    #ploting desired and actual angular velocities of the quadcopter
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    axs[0].plot(t, W_des[0,:], label='Desired Quad Angular Velocity wx')
    axs[0].plot(t, w_quad[0,:], label='Actual Quad Angular Velocity wx')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Angular Velocity wx (rad/s)')
    axs[0].legend()
    axs[1].plot(t, W_des[1,:], label='Desired Quad Angular Velocity wy')
    axs[1].plot(t, w_quad[1,:], label='Actual Quad Angular Velocity wy')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Angular Velocity wy (rad/s)')
    axs[1].legend()
    axs[2].plot(t, W_des[2,:], label='Desired Quad Angular Velocity wz')
    axs[2].plot(t, w_quad[2,:], label='Actual Quad Angular Velocity wz')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Angular Velocity wz (rad/s)')
    axs[2].legend()
    plt.show()
