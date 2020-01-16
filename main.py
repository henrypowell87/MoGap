"""
Author: Henry Powell
Institution: Institute of Neuroscience and Psychology, Glasgow University, Scotland.

High level script to train a denoising autoencoder on a given data set. Different autoencoder models
can be imported from the 'autoencoder_architectures' directory.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils import data
from data_loader import DataSet
from autoencoder_architectures.CNNLSTM import CNNLSTMAE
from functions import load_data, normalize_series, apply_missing, find_translated_mean_pose, find_max_val, filter_tensor

# set script params here
architecture = 'CNNLSTMAE'
# path_to_ground_truth = '/home/henryp/PycharmProjects/MoGap/filtered_tensors'
batch_size = 64
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 90
# pick whether you want to train the network on GPU or CPU
run_on = 'GPU'
# change to false if you want to preload the network with saved weights
train_network = True

# this function finds the mean posture over your whole dataset. Change num_markers to the number of markers
# used for data capture and change central_marker_num to the number of the marker that you would like to
# normalize to the coordinates of (a waist/hip marker is suggested for full body data; collar bone for upper body)
mean_pose = find_translated_mean_pose(num_markers=10,
                                      path='/home/henryp/PycharmProjects/MoGap/ground_truth_tensors_filtered/',
                                      central_marker_num=4, data_type='torch')
# this function finds the max value across you data set for the purposes of -1,1 normalization
max_val = find_max_val(path='/home/henryp/PycharmProjects/MoGap/ground_truth_tensors_filtered/')

# where to save the saved network weights after training
PATH = './MoGapSaveState.pth'

# path to data that has been cropped into smaller same-sized chucks
cropped_ground_truth_path = '/home/henryp/Desktop/cropped_tensors_filtered/'

if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Training on: ' + str(device))

partition, training_set_size = load_data(ground_truth_dir=cropped_ground_truth_path)

training_set = DataSet(list_IDS=partition['train'], data_dir=cropped_ground_truth_path)
testing_set = DataSet(list_IDS=partition['test'], data_dir=cropped_ground_truth_path)

training_generator = data.DataLoader(training_set, **params)
testing_generator = data.DataLoader(testing_set, **params)

# load the network class
# input size is the number of features in your input data
net = CNNLSTMAE(num_frames=64, num_layers=1)
net = net.cuda()

if train_network:

    # select network hyper parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=5, verbose=True)
    # track loss per epoch to plot how the network learns over the number of epochs
    loss_values = []

    # training loop
    for epoch in range(max_epochs):
        for local_batch in training_generator:

            # these code chunks save examples from the training so it is possible to see the change in behvaiour over
            # the training if so desired
            if epoch % 10 == 0:
                local_batch_copy = local_batch.clone()
                torch.save(local_batch_copy, './examples/' + 'original_epoch' + str(epoch) + '.pt')

            # preprocess data
            local_batch = local_batch.float()
            local_batch = normalize_series(local_batch, mean_pose=mean_pose, data_max_val=max_val)
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

            # backwards step, RMSE loss
            loss = criterion(outputs, local_batch)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)

        print('epoch [{}/{}], RMSE_loss:{:.8f}'
              .format(epoch + 1, max_epochs, torch.sqrt(loss.data)))
        loss_values.append(loss.data.item())

    torch.save(net.state_dict(), PATH)

    # plot loss over training
    plt.plot([i for i in range(max_epochs)], loss_values)
    plt.title(architecture + ' loss || Final RMSEloss: ' + str(loss_values[-1]))
    plt.savefig(architecture + '_loss')


else:
    print('Loading saved network state...')
    net.load_state_dict(torch.load(PATH))

# test on testing_set
criterion = nn.MSELoss()
with torch.no_grad():
    for local_batch in testing_generator:
        # preprocess data
        local_batch = local_batch.float()
        local_batch = normalize_series(local_batch, mean_pose=mean_pose, data_max_val=max_val)
        local_batch_missing = apply_missing(time_series_tensor=local_batch, max_erasures=5, max_gap_size=10)

        # move to GPU
        local_batch = local_batch.to(device)
        local_batch_missing = local_batch_missing.to(device)

        outputs = net(local_batch_missing)
        loss = criterion(local_batch, outputs)

    print('Testing RMSELoss:{:.8f}'.format(torch.sqrt(loss.data)))



