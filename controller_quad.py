import numpy as np
def controller(q, traj, quad):
    """
    Explicit controller matching your Python state ordering.

    q    : 8-array or (8,1) column vector with ordering:
           [ yL, zL, dyL, dzL, phiL, dphiL, phiQ, dphiQ ]
           (that's your stated python ordering)
    traj : 6-array [posx, posy, vx, vy, ax, ay]
    quad : quad_w_load_dyn instance (provides mq, ml, l, g, J_quad, ...)
    returns: (f, tau)  where f is total thrust, tau is torque about x
    """

    # make flat 1D array
    q = np.asarray(q).flatten()
    traj = np.asarray(traj).flatten()

    # Gains
    Kpz = 80.0
    Kdz = 20.0
    Kpy = 2.0
    Kdy = 4.8
    KpphiL = 50.0
    KdphiL = 10.0
    KpphiQ = 500.0
    KdphiQ = 30.0
    max_thrust, max_torque = quad.calc_max_torque_thrust()
    min_thrust, min_torque = quad.calc_min_torque_thrust()

    # Unpack python-ordered state
    yL   = q[0]
    zL   = q[1]
    dyL  = q[2]
    dzL  = q[3]
    phiL = q[4]
    dphiL= q[5]
    phiQ = q[6]
    dphiQ= q[7]

    # Unpack trajectory: [posx,posy,vx,vy,ax,ay]
    posx = traj[0]; posy = traj[1]
    vx   = traj[2]; vy   = traj[3]
    ax_des = traj[4]; ay_des = traj[5]

    # total mass
    mQ = quad.mq
    mL = quad.ml
    mT = mQ + mL
    g  = quad.g
    l  = quad.l
    Ixx = quad.J_quad[0,0]

    # --- Load position control (y = posx, z = posy) ---
    e_y = posx - yL
    e_z = posy - zL

    de_y = vx - dyL
    de_z = vy - dzL

    ddy_des = ax_des
    ddz_des = ay_des

    # u1 is total thrust (positive up in z)
    f = mT * (g + ddz_des + Kdz * de_z + Kpz * e_z)

    # desired load acceleration in x (y direction)
    ddy = ddy_des + Kdy * de_y + Kpy * e_y
    phiL_des = -ddy / g

    # --- Load attitude controller ---
    F0 = mT * g

    e_phiL = phiL_des - phiL
    dphiL_des = -(Kpy * de_y) / g
    de_phiL = dphiL_des - dphiL

    ddphiL = KpphiL * e_phiL + KdphiL * de_phiL

    # phiQ desired (quadcopter attitude) (phiQ ~ phiL + coupling)
    phiQ_des = phiL + (mQ * l / F0) * ddphiL

    # --- Quadcopter orientation controller ---
    e_phiQ = phiQ_des - phiQ

    # dphiQ desired must use current dphiL
    dphiQ_des = dphiL + (mQ * l / F0) * (KpphiL * de_phiL)
    de_phiQ = dphiQ_des - dphiQ

    tau = Ixx * (KpphiQ * e_phiQ + KdphiQ * de_phiQ)
    # Saturate
    f = np.clip(f, min_thrust, max_thrust)
    tau = np.clip(tau, min_torque, max_torque)

    return f, tau


def trajectory(t):
    """Generate desired trajectory for the load."""
    # Desired trajectory parameters
    r = 2.0  # radius
    omega = 0.2  # angular velocity
    y_des = lambda t: 5 + r * np.cos(omega * t)
    z_des = lambda t: 5 + r * np.sin(omega * t)
    dy_des = lambda t: -r * omega * np.sin(omega * t)
    dz_des = lambda t: r * omega * np.cos(omega * t)
    ddy_des = lambda t: -r * omega**2 * np.cos(omega * t)
    ddz_des = lambda t: -r * omega**2 * np.sin(omega * t)
    return np.array([
        y_des(t),     # posx
        z_des(t),     # posy
        dy_des(t),    # vx
        dz_des(t),    # vy
        ddy_des(t),   # ax
        ddz_des(t)    # ay
    ])

# def trajectory(t):
#     #straight line trajectory in z direction
#     y_des = 5.0
#     z_des = 5.0 + 0.5 * t
#     dy_des = 0.0
#     dz_des = 0.5    
#     ddy_des = 0.0
#     ddz_des = 0.0
#     return np.array([
#         y_des,     # posx
#         z_des,     # posy
#         dy_des,    # vx
#         dz_des,    # vy
#         ddy_des,   # ax
#         ddz_des    # ay
#     ])

def closed_loop_dynamics(t, x, quad, controller, trajectory):
    """
    This is the exact equivalent of quadLoadDynamics.m
    """

    # 1. Desired trajectory
    traj = trajectory(t)

    # 2. Control
    f, tau = controller(x, traj, quad)

    # 3. System dynamics
    x_dot = quad.dynamics(x.reshape((quad.n_states, 1)), f, tau)

    # 4. Return flat vector (required by integrators)
    return x_dot.flatten()

def closed_loop_dynamics_point(t, x, quad, controller, trajectory):
    """
    This is the exact equivalent of quadLoadDynamics.m
    """

    # 1. Desired trajectory
    

    # 2. Control
    f, tau = controller(x, trajectory, quad)

    # 3. System dynamics
    x_dot = quad.dynamics(x.reshape((quad.n_states, 1)), f, tau)
    

    # 4. Return flat vector (required by integrators)
    return x_dot.flatten()
    