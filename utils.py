import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm

def R3_so3(w):
    v3 = w[2]
    v2 = w[1]
    v1 = w[0]
    so3 = np.array([[ 0 , -v3,  v2],
          [v3,   0, -v1],
          [-v2,  v1,   0]])

    return so3

def so3_R3(Rot):

    log_R = logm(Rot)
    w1 = log_R[2,1]
    w2 = log_R[0,2]
    w3 = log_R[1,0]
    w = np.array([w1,w2,w3]).T
    return w