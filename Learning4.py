'''
Learning 4는 Learning 3의 encoding 방법을 따라 PID 제어기 데이터를 supervised learning 할 계획임
즉 input이 spike train 임
그냥 전체에 대해 학습시켜 보았음
2024-01-04 KDH
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import datetime
import Readfile

dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.cuda.init()
print(f'Current device: {device}')

# NN parms
nb_inputs = 26
nb_hidden = 200
nb_outputs = 4
nb_steps = 300
batch_size = 1000
nb_epochs = 100
lr = 3e-4

# mem and sys params
time_step = 1e-3
tau_mem = 10e-3
tau_syn = 5e-3
alpha = float(np.exp(-time_step / tau_syn))
beta = float(np.exp(-time_step / tau_syn))

# read file
x_tensor, y_tensor = Readfile.f_read_data2()

# mk tensor to array
x_train = np.array(x_tensor, dtype=np.float32)
y_train = np.array(y_tensor, dtype=np.float32)

# weight initalize
weight_scale = 7 * (1.0 - beta)

w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale / np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

w3 = torch.empty((nb_steps, 1), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w3, mean=0.0, std=1)

w4 = torch.empty((nb_steps, 1), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w4, mean=0.0, std=1)

print("init done")


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

def train(x_data, y_data, lr=2e-3, nb_epochs=10):
    params = [w1, w2, w3, w4]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    loss_fn = nn.L1Loss()

    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in spk_train_data_generator(x_data, y_data, batch_size, nb_steps, nb_inputs):
            output, _ = run_snn2(x_local)
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


loss_hist = train(x_train, y_train, lr=lr, nb_epochs=nb_epochs)

def save_to_csv(w1, w2, w3, w4, loss_hist):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"weights_{current_time}_l4_H{nb_hidden}_S{nb_steps}_E{nb_epochs}_R{lr}.csv"

    output_folder = os.path.join('weights')
    os.makedirs(output_folder, exist_ok=True)

    w1_numpy = w1.cpu().detach().numpy()
    w2_numpy = w2.cpu().detach().numpy()
    w3_numpy = w3.cpu().detach().numpy()
    w4_numpy = w4.cpu().detach().numpy()

    df_w1 = pd.DataFrame(w1_numpy, columns=[f'w1_{i}' for i in range(w1.shape[1])])
    df_w2 = pd.DataFrame(w2_numpy, columns=[f'w2_{i}' for i in range(w2.shape[1])])
    df_w3 = pd.DataFrame(w3_numpy, columns=[f'w3_{i}' for i in range(w3.shape[1])])
    df_w4 = pd.DataFrame(w4_numpy, columns=[f'w4_{i}' for i in range(w4.shape[1])])
    df_loss = pd.DataFrame(loss_hist, columns=['Loss'])

    df_all = pd.concat([df_w1, df_w2, df_w3, df_w4, df_loss], axis=1)
    df_all.to_csv(os.path.join(output_folder, file_name), index=False)

    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('loss History')
    plt.savefig(os.path.join(output_folder, f'loss_plot_{current_time}.png'))
    plt.close()

    print("Save Completed!")
    return

# save weights and loss history
save_to_csv(w1, w2, w3, w4, loss_hist)

plt.figure(figsize=(3.3, 2), dpi=150)
plt.plot(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.close()

