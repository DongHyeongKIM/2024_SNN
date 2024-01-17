import numpy as np

m = 0.407
l = 0.1125
Ixx = 0.001421219
Iyy = 0.001560811
Izz = 0.002674212


def euler_to_quaternion(euler):

    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return np.array([q_w, q_x, q_y, q_z])

def quaternion_to_euler(q):
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qw*qy + qx*qz)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)]
    ])
    roll = np.arctan2(R[2,1],R[2,2])
    pitch = np.arcsin(-R[2,0])
    yaw = np.arctan2(R[1,0],R[0,0])

    return np.array([roll, pitch, yaw])

def euler_to_rotation_matrix(euler):

    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # ZYX 회전 순서로 회전 행렬 구성
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

    return rotation_matrix

def quat_to_rotation_matrix(q):
    q_normalized = q / np.linalg.norm(quat)
    qw, qx, qy, qz = q_normalized
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qw*qy + qx*qz)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def rotation_matrix_to_euler_angles(rotation_matrix, radians=True):
    """
    Convert a 3x3 rotation matrix to ZYX Euler angles.
    Args:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix
        radians (bool): If True, return angles in radians. If False, return angles in degrees.
    Returns:
        numpy.ndarray: ZYX Euler angles [roll, pitch, yaw]
    """
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    if sy < 1e-6:
        # Singular case: sy is close to zero
        roll = np.arctan2(rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    else:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    if not radians:
        # Convert angles to degrees
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)

    return np.array([roll, pitch, yaw])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def eta1dot_to_Omega(eta1dot, eta):

    e1 = eta1dot[0]
    e2 = eta1dot[1]
    e3 = eta1dot[2]

    wx = e1 - np.sin(eta[1])*e3
    wy = np.cos(eta[0])*e2 + np.sin(eta[0])*np.cos(eta[1])*e3
    wz = -np.sin(eta[0])*e2 + np.cos(eta[0])*np.cos(eta[1])*e3

    omega = np.array([wx,wy,wz])

    return omega

def odot_p2dot_to_f(omegadot, p2dot, eta):

    sinPhi = np.sin(eta[0])
    sinTheta = np.sin(eta[1])
    cosPhi = np.cos(eta[0])
    cosTheta = np.cos(eta[1])

    dForce = p2dot/(cosTheta*cosPhi)
    fAltitude = (m/4)*dForce*np.array([1,1,1,1])

    fRoll = (Ixx/2*l)*omegadot[0]*np.array([0,-1,0,1])
    fPitch = (Iyy/2*l)*omegadot[1]*np.array([-1,0,1,0])
    tYaw = (Izz / 4)*omegadot[2]*np.array([-1,1,-1,1])

    f = fAltitude + fRoll + fPitch + 78.62778730703259*tYaw

    return f



# quat = [1,1,5,1]
# quat = quat / np.linalg.norm(quat)
#
# euler = quaternion_to_euler(quat)
# print(euler)
# Re = euler_to_rotation_matrix(euler)
# print(Re)
#
# print("----------")
# Rq = quat_to_rotation_matrix(quat)
# print(Rq)
# print("----------------")
# print(rotation_matrix_to_euler_angles(Rq))
# print(rotation_matrix_to_euler_angles(Re))
