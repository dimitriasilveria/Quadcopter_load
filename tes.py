import numpy as np
from scipy.spatial.transform import Rotation

# Global variables to replace MATLAB's persistent variables
class PersistentVars:
    def __init__(self):
        self.hquiver = None
        self.tPrev = None
        self.pcS = None
        self.RcS = None

persistent_vars = PersistentVars()


def hat(v):
    """Skew-symmetric matrix from vector"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def vee(M):
    """Inverse of hat operation - extract vector from skew-symmetric matrix"""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def dynamics_and_control(t, q, trajhandle, params, sdata):
    """
    q = [xL yL zL dxL dyL dzL pL1 pL2 pL3 wL1 wL2 wL3 R11 R21 R31 R12 R22 R32 R13 R23 R33 p q r]
         0  1  2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21 22 23
    """
    global persistent_vars
    
    qDes = trajhandle(t, [])
    
    # Simulation and State Parameters
    mQ = params['massQ']
    mL = params['massL']
    l = params['cableLength']
    g = params['gravity']
    e3 = np.array([0, 0, 1])
    
    xL = q[0:3]
    DxL = q[3:6]
    p = q[6:9]
    wL = q[9:12]
    R = q[12:21].reshape(3, 3, order='F')  # Fortran order for column-major
    wQ = q[21:24]
    b3 = R @ e3  # Vector normal to drone
    
    # Controller initialization
    if t == 0:
        persistent_vars.pcS = np.vstack([p.reshape(1, -1), np.zeros((1, 3))])
        persistent_vars.RcS = np.vstack([q[12:21].reshape(1, -1), np.zeros((1, 9))])
        persistent_vars.tPrev = 0
    
    # Controller Gains - My Parameters
    Kx = 0.5 * np.diag([6, 6, 6.5])
    Kv = 1 * np.diag([6, 6, 4.5])
    Kp = 1 * 9
    KwL = 1.5 * 7.5
    KR = np.diag([1.4, 1.4, 0.4])
    KwQ = np.diag([0.12, 0.12, 0.08])
    
    # Desired State
    xLd = qDes['pos']
    DxLd = qDes['vel']
    DDxLd = qDes['acc']
    
    Dp = np.cross(wL, p)  # OK
    
    # Load Position Controller
    ex = xL - xLd
    ev = DxL - DxLd
    
    A = (-Kx @ ex - Kv @ ev + (mQ + mL) * (DDxLd + g * e3) + 
         mQ * l * (Dp.T @ Dp) * p)
    pc = -A / np.linalg.norm(A)
    
    xpc, dxpc, ddxpc = command_filter(t - persistent_vars.tPrev, 
                                      persistent_vars.pcS, pc, 0.98, 7)
    persistent_vars.pcS = np.vstack([xpc, dxpc])
    
    Dpc = dxpc.flatten()
    DDpc = ddxpc.flatten()
    
    # Load Attitude Controller
    PSI_L = 1 - pc.T @ p
    ep = hat(p) @ hat(p) @ pc
    eDp = Dp - np.cross(np.cross(pc, Dpc), p)
    Fn = (A.T @ p) * p
    Fpd = -Kp * ep - KwL * eDp
    Fff = (mQ * l * np.dot(p, np.cross(pc, Dpc)) * np.cross(p, Dp) + 
           mQ * l * np.cross(np.cross(pc, DDpc), p))
    F = Fn - Fpd - Fff
    b3c = F / np.linalg.norm(F)
    
    b1d = np.array([np.cos(qDes['yaw']), np.sin(qDes['yaw']), 0])
    b1c = (-np.cross(b3c, np.cross(b3c, b1d)) / 
           np.linalg.norm(np.cross(b3c, b1d)))
    b2c = np.cross(b3c, b1c)
    Rc = np.column_stack([b1c, b2c, b3c])
    
    xR, dxR, ddxR = command_filter(t - persistent_vars.tPrev, 
                                   persistent_vars.RcS, Rc.flatten('F'), 0.98, 75)
    persistent_vars.RcS = np.vstack([xR, dxR])
    
    Rce = Rc
    dRce = dxR.reshape(3, 3, order='F')
    ddRce = ddxR.reshape(3, 3, order='F')
    
    wQcHate = Rce.T @ dRce
    DwQcHate = -wQcHate @ wQcHate + Rce.T @ ddRce
    wQce = vee(wQcHate)
    DwQce = vee(DwQcHate)
    
    wQc = wQce
    DwQc = DwQce
    
    # Quadrotor Attitude Controller
    PSI_Q = 0.5 * np.trace(np.eye(3) - Rc.T @ R)
    eR = 0.5 * vee(Rc.T @ R - R.T @ Rc)
    eWQ = wQ - R.T @ Rc @ wQc
    f = F.T @ b3
    M = (-KR @ eR - KwQ @ eWQ + np.cross(wQ, params['I'] @ wQ) - 
         params['I'] @ (hat(wQ) @ R.T @ Rc @ wQc - R.T @ Rc @ DwQc))
    
    # Dynamics
    dxL = DxL
    pL = p
    dpL = Dp
    dwL = -np.cross(pL, f * R @ e3) / (mQ * l)
    ddxL = ((1 / (mQ + mL)) * (pL.T @ f * R @ e3 - mQ * l * (dpL.T @ dpL)) * pL - 
            g * e3)
    dR = R @ hat(wQ)
    dwQ = params['invI'] @ (M - np.cross(wQ, params['I'] @ wQ))
    
    dq = np.concatenate([
        dxL,
        ddxL,
        dpL,
        dwL,
        dR.flatten('F'),
        dwQ
    ])
    
    sdata['F'] = f
    sdata['M'] = M
    sdata['dq'] = dq
    sdata['psiQ'] = PSI_Q
    sdata['psiL'] = PSI_L
    vectorPlot = np.column_stack([b3c, b1d, b1c, b2c, pc]) / 10
    sdata['vectorPlot'] = vectorPlot
    Ftension = np.dot(mL * (ddxL + g * e3), p)
    
    persistent_vars.tPrev = t
    
    return dq


def command_filter(dt, x, uc, zeta, wn):
    """Command filter implementation"""
    state = runge_kutta([], x, uc, zeta, wn, dt)
    dstate = filter_dynamics([], state, uc, zeta, wn)
    x = state[0, :]
    dx = state[1, :]
    ddx = dstate[1, :]
    
    return x, dx, ddx


def filter_dynamics(t, x, uc, zeta, wn):
    """Filter dynamics"""
    dx = np.vstack([
        x[1, :],
        -2 * zeta * wn * x[1, :] - wn**2 * (x[0, :] - uc)
    ])
    return dx


def runge_kutta(t, x, uc, zeta, wn, dt):
    """4th order Runge-Kutta integration"""
    h1 = filter_dynamics(t, x, uc, zeta, wn)
    h2 = filter_dynamics(t + 0.5*dt, x + 0.5*dt*h1, uc, zeta, wn)
    h3 = filter_dynamics(t + 0.5*dt, x + 0.5*dt*h2, uc, zeta, wn)
    h4 = filter_dynamics(t + dt, x + dt*h3, uc, zeta, wn)
    xout = x + dt * (h1 + 2*h2 + 2*h3 + h4) / 6
    
    return xout