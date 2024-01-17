'''
Thrust 가 제대로 학습되었는지 확인하는 코드
'''

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from matplotlib.gridspec import GridSpec
import Readfile

# Flags__________________________________________________________________
Evaluate_Mode = True
Sample_shuffle = False
SHOW_RESULT = True
SAVE_RESULT = False

# parameters___________________________________________________________
nb_inputs = 2
nb_hidden = 100
nb_outputs = 2
nb_steps = 200
############################################
if Evaluate_Mode:
    batch_size = 11000
else:
    batch_size = 1  #1 for sim
###########################################
time_step = 1e-3
tau_mem = 10e-3
tau_syn = 5e-3
alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))
# ________________________________________________________________________
weights_path = 'weights/weights_2024-01-02_21-06-16_l2.csv'
# ________________________________________________________________________

device = torch.device("cpu")
print(device)
dtype = torch.float32

def readfile(weight_file):
    # read data file
    x_tensor, y_tensor = Readfile.f_read_linear(nb_steps)
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
        rst = out.detach()

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
    h2_1 = h2[:,:,0].unsqueeze(-1)
    h2_2 = h2[:,:,1].unsqueeze(-1)
    out1 = torch.einsum("abc,bd->adc", (h2_1, W3))
    out2 = torch.einsum("abc,bd->adc", (h2_2, W4))
    out = torch.cat([out1, out2], dim=-1)

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
    x_data, y_data, W1, W2, W3, B3, W4, v1 = readfile(weights_path)
    x_sample, y_sample = get_sample(x_data,y_data,Sample_shuffle)
    m, other_recordings = run_snn2(x_sample.to_dense())
    m = m.squeeze()
    mem_rec, spk_rec = other_recordings
else:
    W1, W2, W3, B3, W4, v1 = Readfile.f_read_weights(weights_path)

if SHOW_RESULT:
    loss_f = nn.L1Loss()
    loss = loss_f(m, y_sample).item()
    print("loss : ",loss)
    plt.figure(figsize=(10, 6))
    plt.plot(range(batch_size), m[:, :].numpy(), label='y_expected')
    plt.plot(range(batch_size), y_sample[:, :].numpy(), "--", label='y_real')
    plt.xlabel('sample index')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'y loss : {loss}')
    plt.show()


