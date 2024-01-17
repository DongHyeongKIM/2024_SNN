import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import datetime
import Readfile

print(torch.__version__)

# set device for GPU
dtype = torch.float
# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)
torch.cuda.init()

# SNN parameters
nb_inputs = 25
nb_hidden = 150
nb_outputs = 4
time_step = 1e-3
nb_steps = 200
batch_size = 1000

# mem and syn parameters
tau_mem = 10e-3
tau_syn = 5e-3
alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_mem))

# read file
x_tensor, y_tensor = Readfile.f_read_data(nb_steps)
# x_tensor, y_tensor = Readfile.f_read_linear(nb_steps)

# make data to array
x_train = np.array(x_tensor, dtype=np.float32)
y_train = np.array(y_tensor, dtype=np.float32)

# weight initalize
weight_scale = 7 * (1.0 - beta)  # this should give us some spikes to begin with

w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

w3 = torch.empty((nb_steps, 1), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w3, mean=0.0, std=1)

b3 = torch.empty((1, 1), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(b3, mean=0.0, std=1)

w4 = torch.empty((nb_steps, 1), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w4, mean=0.0, std=1)

v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(v1, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

print("init done")


def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True):
    labels_ = np.array(y, dtype=np.float32)
    number_of_batches = len(X) // batch_size
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = 20e-3 / time_step
    firing_times = np.array(X.astype(int))
    unit_numbers = np.arange(nb_units)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            c = firing_times[idx] < nb_steps
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device)

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


#
# def run_snn(inputs):
#
#     #syn : current, mem: membrane potential
#     syn = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
#     mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
#
#     mem_rec = []
#     spk_rec = []
#
#     out = torch.zeros((batch_size,nb_hidden),device=device, dtype=dtype)
#     h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
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
#     h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
#     flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
#     out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
#     out_rec = [out]
#     for t in range(nb_steps):
#         new_flt = alpha * flt + h2[:, t]
#         new_out = beta * out + flt
#
#         flt = new_flt
#         out = new_out
#
#         out_rec.append(out)
#
#     out_rec = torch.stack(out_rec, dim=1)
#     other_recs = [mem_rec, spk_rec]
#     return out_rec, other_recs
#

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

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    h2_1 = h2[:, :, 0].unsqueeze(-1)
    h2_2 = h2[:, :, 1:]
    out1 = torch.einsum("abc,bd->adc", (h2_1, w3))
    out1 = out1 + b3
    out2 = torch.einsum("abc,bd->adc", (h2_2, w4))
    out = torch.cat([out1, out2], dim=-1)

    other_recs = [mem_rec, spk_rec]
    return out, other_recs


def train(x_data, y_data, lr=2e-3, nb_epochs=10):
    params = [w1, w2, w3, b3, w4]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    # loss_fn = nn.NLLLoss()
    loss_fn = nn.L1Loss()

    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in sparse_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs):
            output, _ = run_snn2(x_local.to_dense())
            output = output.squeeze()
            loss_val = loss_fn(output, y_local)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())

        mean_loss = np.mean(local_loss)
        print("Epoch %i: loss=%.5f" % (e + 1, mean_loss))
        loss_hist.append(mean_loss)

    return loss_hist


print(w1, w2, w3, w4, b3)

loss_hist = train(x_train, y_train, lr=3e-4, nb_epochs=150)

print(w1, w2, w3, w4, b3)


def save_to_csv(w1, w2, w3, b3, w4, v1, loss_hist):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"weights_{current_time}.csv"

    output_folder = os.path.join('weights')
    os.makedirs(output_folder, exist_ok=True)

    w1_cpu = w1.cpu().detach()
    w2_cpu = w2.cpu().detach()
    w3_cpu = w3.cpu().detach()
    b3_cpu = b3.cpu().detach()
    w4_cpu = w4.cpu().detach()
    v1_cpu = v1.cpu().detach()

    w1_numpy = w1_cpu.numpy()
    w2_numpy = w2_cpu.numpy()
    w3_numpy = w3_cpu.numpy()
    b3_numpy = b3_cpu.numpy()
    w4_numpy = w4_cpu.numpy()
    v1_numpy = v1_cpu.numpy()

    df_w1 = pd.DataFrame(w1_numpy, columns=[f'w1_{i}' for i in range(w1.shape[1])])
    df_w2 = pd.DataFrame(w2_numpy, columns=[f'w2_{i}' for i in range(w2.shape[1])])
    df_w3 = pd.DataFrame(w3_numpy, columns=[f'w3_{i}' for i in range(w3.shape[1])])
    df_b3 = pd.DataFrame(b3_numpy, columns=[f'b3_{i}' for i in range(b3.shape[1])])
    df_w4 = pd.DataFrame(w4_numpy, columns=[f'w4_{i}' for i in range(w4.shape[1])])
    df_v1 = pd.DataFrame(v1_numpy, columns=[f'v1_{i}' for i in range(v1.shape[1])])
    df_loss = pd.DataFrame(loss_hist, columns=['Loss'])

    df_all = pd.concat([df_w1, df_w2, df_w3, df_b3, df_w4, df_v1, df_loss], axis=1)
    df_all.to_csv(os.path.join(output_folder, file_name), index=False)

    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('loss History')
    plt.savefig(os.path.join(output_folder, f'loss_plot_{current_time}.png'))
    plt.close()
    return


save_to_csv(w1, w2, w3, b3, w4, v1, loss_hist)
print("Save Completed!")

plt.figure(figsize=(3.3, 2), dpi=150)
plt.plot(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.close()
