import numpy as np
import SNN_generator
import utiles

Ixx = 0.001318214634861
Iyy = 0.001443503665657
Izz = 0.002477708981071
Iv = np.array([
    [Ixx, 0, 0],
    [0, Iyy, 0],
    [0, 0, Izz]
])

def PID(state_curr):

    posRef = np.array([0, 0, 0])

    pos = np.array(state_curr[0])
    vel = np.array(state_curr[1])
    quat = np.array(state_curr[2])
    omega = np.array(state_curr[3])

    eta = utiles.quaternion_to_euler(quat)


    mass = 0.407
    grav = 9.8066
    gravVec = np.array([0, 0, -grav])

    rotMat = utiles.euler_to_rotation_matrix(eta)
    rotMat = rotMat


    zBody = rotMat[:,2]

    kp = 1.0
    kv = 0.8
    fDes = kv*(kp*(posRef - pos) - vel) - mass * gravVec

    thrustDes = np.dot(fDes, zBody)

    zBodyDes = fDes / np.linalg.norm(fDes)

    yCenDes= np.array([-np.sin(0), np.cos(0), 0])
    xBodyDes = np.cross(yCenDes, zBodyDes)
    xBodyDes = xBodyDes / np.linalg.norm(xBodyDes)

    yBodyDes = np.cross(zBodyDes, xBodyDes)
    rotMatDes = np.vstack([xBodyDes, yBodyDes, zBodyDes])
    rotMatDes = rotMatDes.T

    kr = 8.0
    kw = 0.05
    eR0 = 0.5 * (rotMat.T @ rotMatDes - rotMatDes.T @ rotMat)

    omegadotdes = kw * (kr * np.array([eR0[2,1],eR0[0,2],eR0[1,0]]) - omega)

    torqureDes = omegadotdes

    return thrustDes, torqureDes

