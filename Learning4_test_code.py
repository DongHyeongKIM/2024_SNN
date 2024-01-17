'''
Learning 4 test 하는 코드
'''
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from matplotlib.gridspec import GridSpec
import Readfile

# Flags__________________________________________________________________
Evaluate_Mode = False
Sample_shuffle = False
SHOW_RESULT = False
SAVE_RESULT = False

# NN parms
nb_inputs = 26
nb_hidden = 200
nb_outputs = 4
nb_steps = 300
############################################
if Evaluate_Mode:
    batch_size = 200
else:
    batch_size = 1  # 1 for sim
###########################################
# mem and sys params
time_step = 1e-3
tau_mem = 10e-3
tau_syn = 5e-3
alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_syn))
# ________________________________________________________________________
weights_path = 'weights/weights_2024-01-08_23-09-19_l4_H200_S300_E100_R0.0003.csv'
device = torch.device("cpu")
print(device)
dtype = torch.float32


def __readfile(weights_path):
    # read data file
    x_tensor, y_tensor = Readfile.f_read_data2()
    # mk tensor to array
    x_train = np.array(x_tensor, dtype=np.float32)
    y_train = np.array(y_tensor, dtype=np.float32)
    # read weight file
    w1, w2, w3, w4 = Readfile.f_read_weights2(weights_path)
    print("File reading complete!")

    return x_train, y_train, w1, w2, w3, w4


def spk_train_data_generator(x, y, batch_size, nb_steps, nb_inputs, shuffle=True):
    labels_ = np.array(y, dtype=np.float32)
    number_of_batches = len(x) // batch_size
    sample_index = np.arange(len(x))

    if shuffle:
        np.random.shuffle(sample_index)

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        x_batch = x[batch_index]
        x_batch = x_batch[:, np.newaxis, :]

        spike_train = np.ones((batch_size, nb_steps, nb_inputs), dtype=np.float32)
        spike_train *= x_batch

        X_batch = torch.tensor(spike_train, device=device)
        Y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch.to(device=device), Y_batch.to(device=device)

        counter += 1


class SurrGradSpike(torch.autograd.Function):
    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


spike_fn = SurrGradSpike.apply


def run_snn2(inputs):
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()

        new_syn = alpha * syn + h1[:, t]
        new_mem = (beta * mem + syn) * (1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out)

        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    # # one hot coding
    # max_ind = torch.argmax(spk_rec, dim=2)
    # remake_spk = torch.zeros_like(spk_rec)
    # for i in range(spk_rec.size(0)):
    #     remake_spk[i, :, :max_ind[i, :].max() + 1] = spk_rec[i, :, :max_ind[i, :].max() + 1]

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))

    h2_1 = h2[:, :, 0].unsqueeze(-1)
    h2_2 = h2[:, :, 1:]
    out1 = torch.einsum("abc,bd->adc", (h2_1, w3))
    out2 = torch.einsum("abc,bd->adc", (h2_2, w4))
    out = torch.cat([out1, out2], dim=-1)
    # out = torch.einsum("abc,bd->adc", (h2, w3))

    other_recs = [mem_rec, spk_rec]
    return out, other_recs


def get_sample(xdata, ydata, shuffle=True):
    for ret in spk_train_data_generator(xdata, ydata, batch_size, nb_steps=nb_steps, nb_inputs=nb_inputs,
                                        shuffle=shuffle):
        return ret


def plot_voltage_traces(mem, spk=None, dim=(2, 2), spike_height=5):
    gs = GridSpec(*dim)
    if spk is not None:
        dat = 1.0 * mem
        dat[spk > 0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[3000])
        ax.axis("off")


if Evaluate_Mode:
    x_data, y_data, w1, w2, w3, w4 = __readfile(weights_path)
    x_sample, y_sample = get_sample(x_data, y_data, Sample_shuffle)
    m, other_recordings = run_snn2(x_sample)
    m = m.squeeze()
    mem_rec, spk_rec = other_recordings
else:
    w1, w2, w3, w4 = Readfile.f_read_weights2(weights_path)

if SHOW_RESULT:
    for i in range(4):
        loss_f = nn.L1Loss()
        loss = [0] * 4
        loss[i] = loss_f(m[:, i], y_sample[:, i]).item()
        print("loss : ", loss[i])
        plt.figure(figsize=(10, 6))
        plt.plot(range(batch_size), m[:, i].numpy(), label='y_expected')
        plt.plot(range(batch_size), y_sample[:, i].numpy(), "--", label='y_real')
        plt.xlabel('sample index')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'loss{i} : {loss[i]}')
        plt.show()

if SAVE_RESULT:
    loss = nn.L1Loss()
    labels = ['Thrust', 'Torque1', 'Torque2', 'Torque3']

    l1_loss = []
    for i in range(4):
        l1_loss_val = loss(m[:, i], y_sample[:, i]).item()
        l1_loss.append(l1_loss_val)
        print(f'L1 loss of {labels[i]}: {l1_loss_val}')

    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    weights_name = weights_path.split('/')[-1]
    output_folder = os.path.join(results_folder, f'{weights_name}_B{batch_size}')
    os.makedirs(output_folder, exist_ok=True)

    for i in range(4):
        plt.figure(figsize=(10, 6))
        plt.plot(range(batch_size), m[:, i].numpy(), label=f'{labels[i]} expected')
        plt.plot(range(batch_size), y_sample[:, i].numpy(), "--", label='GT')
        plt.xlabel('sample index')
        plt.ylabel(f'{labels[i]}')
        plt.legend()
        plt.title(f'{labels[i]} comparison, L1 loss : {l1_loss[i]}')

        output_file_path = os.path.join(output_folder, f'{labels[i]}_results_graph.png')
        plt.savefig(output_file_path)
        plt.show()
        plt.close()

    print("Graph saved!")


#################################################################################################
# funcion for quad sim

def readstate(state_curr):
    pos = np.array(state_curr[0])  # (x , y,  z )
    vel = np.array(state_curr[1])  # (Vx, Vy, Vz)
    quat = np.array(state_curr[2])  # (q0 , qx, qy, qz)
    quat1 = np.array([quat[0]])  # q0
    quat2 = np.array([quat[1], quat[2], quat[3]])  # (qx, qy, qx)
    omega = np.array(state_curr[3])  # (pi, theta, psi)

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

    pos_positive = np.where(pos_pos, pos, 0)
    pos_positive = (pos_positive / 8).clip(0, 1)
    pos_negative = np.where(pos_neg, pos, 0)
    pos_negative = (-pos_negative / 8).clip(0, 1)
    vel_positive = np.where(vel_pos, vel, 0)
    vel_positive = (vel_positive / 14).clip(0, 1)
    vel_negative = np.where(vel_neg, vel, 0)
    vel_negative = (-vel_negative / 14).clip(0, 1)
    quat1_positive = np.where(quat1_pos, quat1, 0)
    quat1_positive = (quat1_positive / 1).clip(0, 1)
    quat1_negative = np.where(quat1_neg, quat1, 0)
    quat1_negative = (-quat1_negative / 1).clip(0, 1)
    quat2_positive = np.where(quat2_pos, quat2, 0)
    quat2_positive = (quat2_positive / 1).clip(0, 1)
    quat2_negative = np.where(quat2_neg, quat2, 0)
    quat2_negative = (-quat2_negative / 1).clip(0, 1)
    omega_positive = np.where(omega_pos, omega, 0)
    omega_positive = (omega_positive / 30).clip(0, 1)
    omega_negative = np.where(omega_neg, omega, 0)
    omega_negative = (-omega_negative / 30).clip(0, 1)

    x_data = np.concatenate(
        (pos_positive, pos_negative,
         vel_positive, vel_negative,quat1_positive,
         quat1_negative, quat2_positive, quat2_negative,
         omega_positive, omega_negative), axis=0)

    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    return x_tensor


def time2spk(X):
    x_tensor = torch.ones((nb_steps, nb_inputs))
    for i, num in enumerate(X):
        x_tensor[:, i] *= num

    return x_tensor


def SNN(state_curr):
    state_tensor = readstate(state_curr)
    state_spk_train = time2spk(state_tensor)
    state_spk_train = state_spk_train.unsqueeze(0)
    output, _ = run_snn2(state_spk_train)

    output = output.squeeze()
    output = np.array(output)

    output[1:] = output[1:] / 10.0

    return output
