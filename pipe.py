import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from BdRNNAE import BdRNNAE
from functions import normalize_time_series, split, merge, gap_fill

np.set_printoptions(threshold=sys.maxsize)

file_path = '/home/henryp/PycharmProjects/MoGap/augmented_data/A001P001T014GAPS.csv'
network_weights_path = './MoGapSaveState.pth'
run_on = 'GPU'

if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

net = BdRNNAE(input_size=30, hidden_size=10, num_layers=2)
net = net.double()
net = net.cuda()
net.load_state_dict(torch.load(network_weights_path))

original_data, split_data = split(data=file_path, index_cols=True, size=30, gap_length=5, result_dim=(0, 30, 30))
original_data[original_data == 0.0000] = np.nan
original_data = torch.Tensor(original_data).double()
original_data = normalize_time_series(original_data, min_val=-1, max_val=1)

estimates = np.empty((0, 30, 30))
for slice in split_data:
    slice = torch.Tensor(slice).unsqueeze(0).double()
    slice = normalize_time_series(slice, min_val=-1, max_val=1)
    slice = slice.to(device)

    estimate = net(slice)
    estimate = torch.Tensor.cpu(estimate)
    estimate = torch.detach(estimate).numpy()
    estimates = np.append(estimates, estimate, axis=0)

estimated = merge(estimates, gap_length=5, result_dim=(0, 30))
original_data = original_data.numpy()
gap_filled = gap_fill(original_data, estimated)

plt.plot(original_data)
plt.show()

plt.plot(gap_filled)
plt.show()




