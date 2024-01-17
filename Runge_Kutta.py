import numpy as np

dt = 0.01

def _pos_calculator(X,a):

    k1 = np.array([X[1],a])

    k2 = np.array([X[1] + dt/2. * k1[1], a])

    k3 = np.array([X[1] + dt/2. * k2[1], a])
    k4 = np.array([X[1] + dt * k3[1], a])
    Xs = X + dt/6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return Xs


def __quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def RK(p2dot, omegadot, state_curr):

    pos = np.array(state_curr[0])
    vel = np.array(state_curr[1])
    quat = np.array(state_curr[2])
    omega = np.array(state_curr[3])
    omegadot = np.array(omegadot)

    # pos, vel 구하기
    X = np.zeros([2,3])
    X[0,:] = pos
    X[1,:] = vel

    Xd = _pos_calculator(X, p2dot)
    pd = Xd[0]
    veld = Xd[1]

    omegad = omega + omegadot * dt
    omegad_quat = np.array([0,omegad[0],omegad[1],omegad[2]])
    quatdot = 0.5 * __quat_multiply(quat,omegad_quat)
    quatd = quat + quatdot * dt
    quatd = quatd / np.linalg.norm(quatd)

    return pd, veld, quatd, omegad


# p = [0,0,0]
# vel = [0,0,0]
# quat = [0,0,0,0]
# omega = [0,0,0]
# output = [0,0,0,0]
#
# state_curr = [p,vel,quat,omega]
#
# p2dot = [0,0,1]
# omegadot = [1,0,3]
#
# pd , veld, quatd, omegad = RK(p2dot, omegadot,  state_curr)
# print(pd, veld, quatd, omegad)

