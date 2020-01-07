"""
Author: Henry Powell
Institution: Institute of Neuroscience and Psychology, Glasgow University, Scotland.

Pipe.py provides a general purpose pipeline for performing gap filling on data with missing markers. Make sure that
the network has been trained on data sufficiently similar to the data you are using (same number of markers, same
range of movements etc).
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from autoencoder_architectures.BdLSTMAE import BdLSTMAE
from functions import normalize_series, find_translated_mean_pose, find_max_val, split, merge, gap_fill
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(profile="full")

# Paste the path to the mo cap file you wish to gap fill here
file_path = '/home/henryp/PycharmProjects/MoGap/augmented_data/A001P001T028GAPS.csv'
network_weights_path = './MoGapSaveState.pth'
run_on = 'GPU'

if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

net = BdLSTMAE(input_size=30, hidden_size=200, num_layers=1)
net = net.cuda()
net.load_state_dict(torch.load(network_weights_path))

mean_pose = find_translated_mean_pose(num_markers=10, path='/home/henryp/PycharmProjects/MoGap/ground_truth_data',
                                      central_marker_num=4)

max_val = find_max_val(path='/home/henryp/PycharmProjects/MoGap/ground_truth_data')


original_data, split_data = split(data=file_path, index_cols=True, size=200, padding=1, result_dim=(0, 200, 30))
original_data[original_data == 0.0000] = np.nan
original_data = torch.Tensor(original_data)
original_data = normalize_series(original_data, mean_pose=mean_pose, data_max_val=max_val)

estimates = np.empty((0, 200, 30))
criterion = nn.MSELoss()
for window in split_data:
    window = torch.Tensor(window).unsqueeze(0)
    window = window.float()
    window = normalize_series(window, mean_pose=mean_pose, data_max_val=max_val)
    window = window.to(device)

    estimate = net(window)
    estimate = torch.Tensor.cpu(estimate)
    estimate = torch.detach(estimate).numpy()
    estimates = np.append(estimates, estimate, axis=0)

estimated = merge(estimates, padding=1, result_dim=(0, 30))
original_data = original_data.numpy()

gap_filled = gap_fill(original_data, estimated)

#
np.save('A001P001T028_gaps', original_data)
np.save('A001P001T028_gap_filled', gap_filled)
np.save('A001P001T028_estimated', estimated)

