import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
from data_loader import DataSet
from autoencoder_architectures.BdLSTMAE import BdLSTMAE
from functions import load_data, normalize_time_series, apply_missing

architecture = 'BdLSTMAE'
batch_size = 32
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 90
run_on = 'GPU'
train_network = True

PATH = './MoGapSaveState.pth'

ground_truth_path = '/home/henryp/PycharmProjects/MoGap/cropped_ground_truth_data/'

if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Training on: ' + str(device))

partition, training_set_size = load_data(ground_truth_dir=ground_truth_path)

training_set = DataSet(list_IDS=partition['train'], data_dir=ground_truth_path)
testing_set = DataSet(list_IDS=partition['test'], data_dir=ground_truth_path)

training_generator = data.DataLoader(training_set, **params)
testing_generator = data.DataLoader(testing_set, **params)

net = BdLSTMAE(input_size=30, hidden_size=40, num_layers=2)
net = net.double()
net = net.cuda()

if train_network:

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=5, verbose=True)

    loss_values = []

    for epoch in range(max_epochs):
        for local_batch in training_generator:

            if epoch % 10 == 0:
                local_batch = local_batch.clone()
                torch.save(local_batch, './examples/' + 'original_epoch' + str(epoch) + '.pt')

            # preprocess data
            local_batch = normalize_time_series(local_batch, -1, 1)
            local_batch_missing = apply_missing(time_series_tensor=local_batch, max_erasures=5, max_gap_size=10)

            # move to GPU
            local_batch = local_batch.to(device)
            local_batch_missing = local_batch_missing.to(device)

            if epoch % 10 == 0:
                missing = local_batch_missing.clone()
                torch.save(missing, './examples/' + 'missing_epoch' + str(epoch) + '.pt')

            # training loop
            optimizer.zero_grad()
            outputs = net(local_batch_missing)

            if epoch % 10 == 0:
                estimated = outputs.clone()
                torch.save(estimated, './examples/' + 'estimated_epoch' + str(epoch) + '.pt')

            loss = criterion(outputs, local_batch)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)

        print('epoch [{}/{}], loss:{:.6f}'
              .format(epoch + 1, max_epochs, loss.data.item()))
        loss_values.append(loss.data.item())
    torch.save(net.state_dict(), PATH)

    # plot loss over training
    plt.plot([i for i in range(max_epochs)], loss_values)
    plt.title(architecture + ' loss || Final loss: ' + str(np.mean(loss_values)))
    plt.savefig(architecture + '_loss')


else:
    print('Loading saved network state...')
    net.load_state_dict(torch.load(PATH))

# test on testing_set
running_loss = []
criterion = nn.MSELoss()
with torch.no_grad():
    for local_batch in testing_generator:
        # preprocess data
        local_batch = normalize_time_series(local_batch, -1, 1)
        local_batch_missing = apply_missing(time_series_tensor=local_batch, max_erasures=5, max_gap_size=10)

        # move to GPU
        local_batch = local_batch.to(device)
        local_batch_missing = local_batch_missing.to(device)

        outputs = net(local_batch_missing)
        loss = criterion(local_batch, outputs)
        running_loss.append(loss.data.item())

    print('Testing Loss:{:.4f}'.format(loss.data.item()))



