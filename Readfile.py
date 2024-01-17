import torch
import pandas as pd

def f_read_data(nb_steps):
    csv_file_path = 'data2.csv'
    data_frame = pd.read_csv(csv_file_path)

    pos = data_frame.iloc[:, :3].copy()
    vel = data_frame.iloc[:, 3:6].copy()
    quat1 = data_frame.iloc[:, 6:7].copy()
    quat2 = data_frame.iloc[:, 7:10].copy()
    omega = data_frame.iloc[:, 10:13].copy()
    thrust = data_frame.iloc[:, 13:14].copy()
    torque = data_frame.iloc[:, 14:].copy()

    pos_pos = pos >= 0
    pos_neg = pos < 0
    vel_pos = vel >= 0
    vel_neg = vel < 0
    quat2_pos = quat2 >= 0
    quat2_neg = quat2 < 0
    omega_pos = omega >= 0
    omega_neg = omega < 0

    pos_positive = pos[pos_pos].fillna(0)
    pos_positive = (pos_positive * (-199.5 / 8) + nb_steps-1).clip(0, nb_steps)
    pos_negative = pos[pos_neg].fillna(0)
    pos_negative = (-pos_negative * (-199.5 / 8) + nb_steps-1).clip(0, nb_steps)
    vel_positive = vel[vel_pos].fillna(0)
    vel_positive = (vel_positive * (-199.5 / 14) + nb_steps-1).clip(0, nb_steps)
    vel_negative = vel[vel_neg].fillna(0)
    vel_negative = (-vel_negative * (-199.5 / 14) + nb_steps-1).clip(0, nb_steps)
    quat1 = (quat1 * (-199.5 / 1) + nb_steps-1).clip(0, nb_steps)
    quat2_positive = quat2[quat2_pos].fillna(0)
    quat2_positive = (quat2_positive * (-199.5 / 1) + nb_steps-1).clip(0, nb_steps)
    quat2_negative = quat2[quat2_neg].fillna(0)
    quat2_negative = (-quat2_negative * (-199.5 / 1) + nb_steps-1).clip(0, nb_steps)
    omega_positive = omega[omega_pos].fillna(0)
    omega_positive = (omega_positive * (-199.5 / 55) + nb_steps-1).clip(0, nb_steps)
    omega_negative = omega[omega_neg].fillna(0)
    omega_negative = (-omega_negative * (-199.5 / 55) + nb_steps-1).clip(0, nb_steps)


    x_data = pd.concat(
        [pos_positive, pos_negative, vel_positive, vel_negative, quat1, quat2_positive, quat2_negative, omega_positive,
         omega_negative], axis=1)
    y_data = pd.concat([thrust, torque], axis=1)

    x_tensor = torch.tensor(x_data.values, dtype=torch.int)
    y_tensor = torch.tensor(y_data.values, dtype=torch.float32)

    print("state data size :", x_tensor.size())
    print("target data size :", y_tensor.size())

    return x_tensor, y_tensor

def f_read_data2():
    csv_file_path = 'data4.csv'
    data_frame = pd.read_csv(csv_file_path)

    pos = data_frame.iloc[:, :3].copy()
    vel = data_frame.iloc[:, 3:6].copy()
    quat1 = data_frame.iloc[:, 6:7].copy()
    quat2 = data_frame.iloc[:, 7:10].copy()
    omega = data_frame.iloc[:, 10:13].copy()
    thrust = data_frame.iloc[:, 13:14].copy()
    torque = data_frame.iloc[:, 14:].copy()

    pos_pos = pos >= 0
    pos_neg = pos < 0
    vel_pos = vel >= 0
    vel_neg = vel < 0
    quat1_pos = quat1 >= 0
    quat1_neg = quat1 < 0
    quat2_pos = quat2 >= 0
    quat2_neg = quat2 < 0
    omega_pos = omega >= 0
    omega_neg = omega < 0

    torque = torque * 10

    pos_positive = pos[pos_pos].fillna(0)
    pos_positive = (pos_positive / 8).clip(0, 1)
    pos_negative = pos[pos_neg].fillna(0)
    pos_negative = (-pos_negative / 8).clip(0, 1)
    vel_positive = vel[vel_pos].fillna(0)
    vel_positive = (vel_positive / 14).clip(0, 1)
    vel_negative = vel[vel_neg].fillna(0)
    vel_negative = (-vel_negative / 14).clip(0, 1)
    quat1_positive = quat1[quat1_pos].fillna(0)
    quat1_positive = (quat1_positive / 1).clip(0, 1)
    quat1_negative = quat1[quat1_neg].fillna(0)
    quat1_negative = (-quat1_negative / 1).clip(0, 1)
    quat2_positive = quat2[quat2_pos].fillna(0)
    quat2_positive = (quat2_positive / 1).clip(0, 1)
    quat2_negative = quat2[quat2_neg].fillna(0)
    quat2_negative = (-quat2_negative / 1).clip(0, 1)
    omega_positive = omega[omega_pos].fillna(0)
    omega_positive = (omega_positive / 30).clip(0, 1)
    omega_negative = omega[omega_neg].fillna(0)
    omega_negative = (-omega_negative / 30).clip(0, 1)


    x_data = pd.concat(
        [pos_positive, pos_negative, vel_positive, vel_negative, quat1_positive, quat1_negative,
         quat2_positive, quat2_negative, omega_positive, omega_negative], axis=1)
    y_data = pd.concat([thrust, torque], axis=1)

    x_tensor = torch.tensor(x_data.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_data.values, dtype=torch.float32)

    print("state data size :", x_tensor.size())
    print("target data size :", y_tensor.size())

    return x_tensor, y_tensor



def f_read_linear(nb_steps):
    csv_file_path = 'test_data/y_2x_data.csv'
    data_frame = pd.read_csv(csv_file_path)
    x = data_frame.iloc[:, :1].copy()
    pos_x = x >= 0
    neg_x = x < 0

    x_positive = x[pos_x].fillna(0)
    x_positive = (x_positive * (-199.5 / 7) + nb_steps).clip(0, nb_steps)
    x_negative = x[neg_x].fillna(0)
    x_negative = (-x_negative * (-199.5 / 7) + nb_steps).clip(0, nb_steps)

    x_data = pd.concat([x_positive, x_negative], axis=1)
    # x_data = (x * (-199.5/7)+nb_steps).clip(0,nb_steps)

    y_data = data_frame.iloc[:, 1:].copy()
    y_data = y_data

    x_tensor = torch.tensor(x_data.values, dtype=torch.int)
    y_tensor = torch.tensor(y_data.values, dtype=torch.float32)

    print("x data size :", x_tensor.size())
    print("y data size :", y_tensor.size())

    return x_tensor, y_tensor


def f_read_linear2():
    csv_file_path = 'test_data/y_2x_data.csv'
    data_frame = pd.read_csv(csv_file_path)
    x = data_frame.iloc[:,:1].copy()
    pos_x = x >= 0
    neg_x = x < 0

    x_pos = x[pos_x].fillna(0)
    x_neg = x[neg_x].fillna(0)
    x_pos = (x_pos / 7).clip(0, 1)
    x_neg = (-x_neg / 7).clip(0, 1)
    x_data = pd.concat([x_pos, x_neg], axis=1)

    y_data = data_frame.iloc[:, 1:].copy()

    x_tensor = torch.tensor(x_data.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_data.values, dtype=torch.float32)

    print("x data size:", x_tensor.size())
    print("y data size:", y_tensor.size())

    return x_tensor, y_tensor




def f_read_weights(weight_file):
    weight_file_path = weight_file
    df = pd.read_csv(weight_file_path)

    W1_columns = [col for col in df.columns if col.startswith('w1_')]
    W1 = torch.tensor(df[W1_columns].dropna().values).to(torch.float32)
    W2_columns = [col for col in df.columns if col.startswith('w2_')]
    W2 = torch.tensor(df[W2_columns].dropna().values).to(torch.float32)
    W3_columns = [col for col in df.columns if col.startswith('w3_')]
    W3 = torch.tensor(df[W3_columns].dropna().values).to(torch.float32)
    B3_columns = [col for col in df.columns if col.startswith('b3_')]
    B3 = torch.tensor(df[B3_columns].dropna().values).to(torch.float32)
    W4_columns = [col for col in df.columns if col.startswith('w4_')]
    W4 = torch.tensor(df[W4_columns].dropna().values).to(torch.float32)
    v1_columns = [col for col in df.columns if col.startswith('v1_')]
    V1 = torch.tensor(df[v1_columns].dropna().values).to(torch.float32)

    # print("W1:")
    # print(W1)
    # print("W2:")
    # print(W2)
    # print("W3:")
    # print(W3)
    # print("v1:")
    # print(v1)

    return W1,W2,W3,B3,W4,V1

def f_read_weights2(weight_file):
    weight_file_path = weight_file
    df = pd.read_csv(weight_file_path)

    W1_columns = [col for col in df.columns if col.startswith('w1_')]
    W1 = torch.tensor(df[W1_columns].dropna().values).to(torch.float32)
    W2_columns = [col for col in df.columns if col.startswith('w2_')]
    W2 = torch.tensor(df[W2_columns].dropna().values).to(torch.float32)
    W3_columns = [col for col in df.columns if col.startswith('w3_')]
    W3 = torch.tensor(df[W3_columns].dropna().values).to(torch.float32)
    W4_columns = [col for col in df.columns if col.startswith('w4_')]
    W4 = torch.tensor(df[W4_columns].dropna().values).to(torch.float32)

    return W1,W2,W3,W4