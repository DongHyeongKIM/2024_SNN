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

# parameters___________________________________________________________
nb_inputs = 25
nb_hidden = 150
nb_outputs = 4
time_step = 1e-3
nb_steps = 200
############################################
if Evaluate_Mode:
    batch_size = 200
else:
    batch_size = 1  #1 for sim
###########################################
tau_mem = 10e-3
tau_syn = 5e-3
alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))
# ________________________________________________________________________
weights_path = 'weights/weights_2023-12-12_17-53-48.csv'
# ________________________________________________________________________

device = torch.device("cpu")
print(device)
dtype = torch.float32

def readfile(weight_file):
    # read data file
    x_tensor, y_tensor = Readfile.f_read_data(nb_steps)
    #x_tensor, y_tensor = Readfile.f_read_linear(nb_steps)

    # make data to array
    x_train = np.array(x_tensor, dtype=np.float32)
    y_train = np.array(y_tensor, dtype=np.float32)

    # read weight file
    W1, W2, W3, B3, W4, v1 = Readfile.f_read_weights(weight_file)

    print("File reading complete!")

    return x_train, y_train, W1, W2, W3, B3, W4, v1

def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True ):
    labels_ = np.array(y,dtype=np.float32)
    number_of_batches = len(X)//batch_size
    sample_index = np.arange(len(X))


    firing_times = np.array(X.astype(int))
    unit_numbers = np.arange(nb_units)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:

        if shuffle:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        else:
            rand_num = np.random.randint(number_of_batches)
            batch_index = sample_index[batch_size*rand_num:batch_size*(rand_num+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            c = firing_times[idx]<nb_steps
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index],device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

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

# def run_snn(inputs):
#
#     syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
#     mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
#
#     mem_rec = []
#     spk_rec = []
#
#     out = torch.zeros((batch_size,nb_hidden),device=device, dtype=dtype)
#     h1_from_input = torch.einsum("abc,cd->abd", (inputs, W1))
#     # Compute hidden layer activity
#     for t in range(nb_steps):
#         h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac",(out, v1))
#         mthr = mem - 1.0
#         out = spike_fn(mthr)
#         rst = out.detach()  # We do not want to backprop through the reset
#
#         new_syn = alpha * syn + h1
#         new_mem = (beta * mem + syn) * (1.0 - rst)
#
#         mem_rec.append(mem)
#         spk_rec.append(out)
#
#         mem = new_mem
#         syn = new_syn
#
#     mem_rec = torch.stack(mem_rec, dim=1)
#     spk_rec = torch.stack(spk_rec, dim=1)
#
#     # Readout layer
#     h2 = torch.einsum("abc,cd->abd", (spk_rec, W2))
#     out = torch.einsum("abc,bd->adc",(h2,W3))
#     # flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
#     # out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
#     # out_rec = [out]
#     # for t in range(nb_steps):
#     #     new_flt = alpha * flt + h2[:, t]
#     #     new_out = beta * out + flt
#     #
#     #     flt = new_flt
#     #     out = new_out
#     #
#     #     out_rec.append(out)
#     #
#     # out_rec = torch.stack(out_rec, dim=1)
#     other_recs = [mem_rec, spk_rec]
#     return out, other_recs

def run_snn2(inputs):
    h1 = torch.einsum("abc,cd->abd", (inputs, W1))
    syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    for t in range(nb_steps):
        mthr = mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()  # We do not want to backprop through the reset

        new_syn = alpha * syn + h1[:, t]
        new_mem = (beta * mem + syn) * (1.0 - rst)

        mem_rec.append(mem)
        spk_rec.append(out)

        mem = new_mem
        syn = new_syn

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, W2))
    flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    h2_1 = h2[:, :, 0].unsqueeze(-1)
    h2_2 = h2[:, :, 1:]
    out1 = torch.einsum("abc,bd->adc", (h2_1, W3))
    out1 = out1 + B3
    out2 = torch.einsum("abc,bd->adc", (h2_2, W4))
    out = torch.cat([out1, out2], dim=-1)


    # out_rec = [out]
    # for t in range(nb_steps):
    #     new_flt = alpha * flt + h2[:, t]
    #     new_out = beta * out + flt
    #
    #     flt = new_flt
    #     out = new_out
    #
    #     out_rec.append(out)
    #
    # out_rec = torch.stack(out_rec, dim=1)
    other_recs = [mem_rec, spk_rec]
    return out, other_recs

def get_sample(xdata,ydata,shuffle=False):
    for ret in sparse_data_generator(xdata,ydata,batch_size, nb_steps=nb_steps, nb_units=nb_inputs,shuffle=shuffle):
        return ret

def plot_voltage_traces(mem, spk=None, dim=(2,2), spike_height=5):
    gs=GridSpec(*dim)
    if spk is not None:
        dat = 1.0*mem
        dat[spk>0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[3000])
        ax.axis("off")



if Evaluate_Mode:
    x_data, y_data , W1, W2, W3, B3, W4, v1 = readfile(weights_path)
    x_sample, y_sample = get_sample(x_data,y_data,Sample_shuffle)
    m, other_recordings = run_snn2(x_sample.to_dense())
    m = m.squeeze(1)
    mem_rec, spk_rec = other_recordings
else:
    W1, W2, W3, B3, W4, v1 = Readfile.f_read_weights(weights_path)


#show the results

if SHOW_RESULT:
    loss = nn.functional.smooth_l1_loss(m[:,0],y_sample[:,0]).item()
    print("loss : ",loss)
    plt.figure(figsize=(10, 6))
    plt.plot(range(batch_size), m[:, 0].numpy(), label='y_expected')
    plt.plot(range(batch_size), y_sample[:, 0].numpy(), "--", label='y_real')
    plt.xlabel('sample index')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'y loss : {loss}')
    plt.show()


#save results
if SAVE_RESULT:
    loss = nn.L1Loss()
    mse_thrust_values = loss(m[:,0],y_sample[:,0]).item()
    print("MSE of thrust : ",mse_thrust_values)
    mse_torque1_values = loss(m[:,1],y_sample[:,1]).item()
    mse_torque2_values = loss(m[:,2],y_sample[:,2]).item()
    mse_torque3_values = loss(m[:,3],y_sample[:,3]).item()
    mean_torque_value = torch.mean(torch.tensor([mse_torque1_values,mse_torque2_values,mse_torque3_values])).item()
    print("MSE of torque : ",mean_torque_value)

    results_folder = 'results'
    os.makedirs(results_folder,exist_ok=True)

    weights_name = weights_path.split('/')[-1]
    output_folder = os.path.join(results_folder, f'{weights_name}_B{batch_size}')
    os.makedirs(output_folder,exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.plot(range(batch_size),m[:,0].numpy(), label='thrust expected')
    plt.plot(range(batch_size),y_sample[:,0].numpy(), "--",label='real thrust')
    plt.xlabel('sample index')
    plt.ylabel('Thrust')
    plt.legend()
    plt.title(f'Thrust comparison, L1 loss : {mse_thrust_values}')

    output_file_path1 = os.path.join(output_folder,'Thrust_results_graph.png')
    plt.savefig(output_file_path1)
    plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(range(batch_size),m[:,1].numpy(),label='torqueX expected')
    plt.plot(range(batch_size),y_sample[:,1].numpy(),"--",label='real torque')
    plt.xlabel('sample index')
    plt.ylabel('TorqueX')
    plt.legend()
    plt.title(f'TorqueX comparison, L1 loss : {mse_torque1_values}')

    output_file_path2 = os.path.join(output_folder,'TorqueX_results_graph.png')
    plt.savefig(output_file_path2)
    plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(range(batch_size),m[:,2].numpy(),label='torqueY expected')
    plt.plot(range(batch_size),y_sample[:,2].numpy(),"--",label='real torque')
    plt.xlabel('sample index')
    plt.ylabel('TorqueY')
    plt.legend()
    plt.title(f'TorqueY comparison, L1 loss : {mse_torque2_values}')

    output_file_path3 = os.path.join(output_folder,'TorqueY_results_graph.png')
    plt.savefig(output_file_path3)
    plt.show()
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(range(batch_size),m[:,3].numpy(),label='torqueZ expected')
    plt.plot(range(batch_size),y_sample[:,3].numpy(),"--",label='real torque')
    plt.xlabel('sample index')
    plt.ylabel('TorqueZ')
    plt.legend()
    plt.title(f'TorqueZ comparison, L1 loss : {mse_torque3_values}')

    output_file_path4 = os.path.join(output_folder,'TorqueZ_results_graph.png')
    plt.savefig(output_file_path4)
    plt.show()
    plt.close()
    print(f'graph saved!')




####################################################################################################
# function for quad simulation
def readstate(state_curr):
    pos = np.array(state_curr[0])                                       # (x , y,  z )
    vel = np.array(state_curr[1])                                       # (Vx, Vy, Vz)
    quat = np.array(state_curr[2])                                      # (q0 , qx, qy, qz)
    quat1 = np.array([quat[0]])                                         # q0
    quat2 = np.array([quat[1],quat[2],quat[3]])                         # (qx, qy, qx)
    omega = np.array(state_curr[3])                                     # (pi, theta, psi)

    pos_pos = pos >= 0
    pos_neg = pos < 0
    vel_pos = vel >= 0
    vel_neg = vel < 0
    quat2_pos = quat2 >= 0
    quat2_neg = quat2 < 0
    omega_pos = omega >= 0
    omega_neg = omega < 0

    pos_positive = np.where(pos_pos, pos, 0)
    pos_positive = (pos_positive * (-199.5 / 8) + nb_steps).clip(0, nb_steps)
    pos_negative = np.where(pos_neg, pos, 0)
    pos_negative = (-pos_negative * (-199.5 / 8) + nb_steps).clip(0, nb_steps)
    vel_positive = np.where(vel_pos,vel,0)
    vel_positive = (vel_positive * (-199.5 / 14) + nb_steps).clip(0, nb_steps)
    vel_negative = np.where(vel_neg,vel,0)
    vel_negative = (-vel_negative * (-199.5 / 14) + nb_steps).clip(0, nb_steps)
    quat1 = (quat1 * (-199.5 / 1) + nb_steps).clip(0, nb_steps)
    quat2_positive = np.where(quat2_pos,quat2,0)
    quat2_positive = (quat2_positive * (-199.5 / 1) + nb_steps).clip(0, nb_steps)
    quat2_negative = np.where(quat2_neg,quat2,0)
    quat2_negative = (-quat2_negative * (-199.5 / 1) + nb_steps).clip(0, nb_steps)
    omega_positive = np.where(omega_pos,omega,0)
    omega_positive = (omega_positive * (-199.5 / 55) + nb_steps).clip(0, nb_steps)
    omega_negative = np.where(omega_neg,omega,0)
    omega_negative = (-omega_negative * (-199.5 / 55) + nb_steps).clip(0, nb_steps)


    x_data = np.concatenate(
        (pos_positive, pos_negative,
                vel_positive, vel_negative,
                quat1, quat2_positive, quat2_negative,
                omega_positive,omega_negative), axis=0)

    x_tensor = torch.tensor(x_data, dtype=torch.int)

    return x_tensor


def time2spk(X):
    x_tensor = torch.zeros((nb_steps, nb_inputs))
    for i, num in enumerate(X):
        if num < nb_steps:
            x_tensor[num, i] = 1
    return x_tensor


def SNN(state_curr):

    state_tensor = readstate(state_curr)

    state_spk = time2spk(state_tensor)
    state_spk = state_spk.unsqueeze(0)
    output, _ = run_snn2(state_spk)

    output = output.squeeze()                  # (1,1,4) tensor to (4)
    output = np.array(output)                           # (4,) np array

    return output



