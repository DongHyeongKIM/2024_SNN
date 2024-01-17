import numpy as np


#drone parameter
mass = 0.407           # mass
grav = 9.8066       # gravity
gravVec = np.array([0,0,-grav])

Ixx = 0.001318214634861
Iyy = 0.001443503665657
Izz = 0.002477708981071
Iv = np.array([
    [Ixx, 0, 0],
    [0, Iyy, 0],
    [0, 0, Izz]
])



def Quad(Thrust, Torque, state_curr):

    Thrust = np.array(Thrust)
    Torque = np.array(Torque)

    pos = np.array(state_curr[0])
    vel = np.array(state_curr[1])
    quat = np.array(state_curr[2])
    omega = np.array(state_curr[3])

    q_normalized = quat / np.linalg.norm(quat)
    qw, qx, qy, qz = q_normalized
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qw*qy + qx*qz)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)]
    ])


    #postion
    p2dot = (R@(Thrust.T)) + mass * gravVec             # (3,) size nparray
    p2dot = p2dot / mass

    #angular
    inv_Iv = np.linalg.inv(Iv)
    omegadot = inv_Iv@(Torque.T) - inv_Iv@(np.cross(omega, Iv@omega))

    return p2dot, omegadot




