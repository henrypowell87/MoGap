import torch
import torch.nn as nn
from torch.utils import data
from classes import DataSet, RNNAE
from functions import load_data, normalize_time_series, apply_missing, nan_to_zero

batch_size = 32
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 90
run_on = 'GPU'
train_network = True

ground_truth_path = '/home/henryp/PycharmProjects/MoGap/cropped_ground_truth_data/'

if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Training on: ' + str(device))

partition, training_set_size = load_data(ground_truth_dir=ground_truth_path)

training_set = DataSet(list_IDS=partition['train'], data_dir=ground_truth_path)
training_generator = data.DataLoader(training_set, **params)

net = RNNAE(input_size=30, hidden_size=20, num_layers=5)
net = net.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    net.parameters(), lr=1e-6, weight_decay=1e-5
)

for epoch in range(max_epochs):
    for local_batch in training_generator:

        # preprocess data
        local_batch = normalize_time_series(local_batch, -1, 1)
        local_batch_missing = apply_missing(time_series_tensor=local_batch, max_erasures=5, max_gap_size=10)

        # move to GPU
        local_batch = local_batch.to(device)
        local_batch_missing = local_batch_missing.to(device)

        # training loop
        outputs = net(local_batch_missing)
        loss = criterion(outputs, local_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, max_epochs, loss.data.item()))


