"""
Author: Henry Powell
Institution: Institute of Neuroscience and Psychology, Glasgow University, Scotland.

High level script to train a denoising autoencoder on a given data set. Different autoencoder models
can be imported from the 'autoencoder_architectures' directory.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from data_loader import DataSet
from RMSE_loss import RMSE_loss
from autoencoder_architectures.LSTMAE import LSTMAE
from functions import load_data, apply_missing_cmu_val

# set script params here
architecture = 'LSTMAE'
# path_to_ground_truth = '/home/henryp/PycharmProjects/MoGap/filtered_tensors'
batch_size = 32
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 90
# pick whether you want to train the network on GPU or CPU
run_on = 'GPU'
# change to false if you want to preload the network with saved weights
train_network = True

# get mean pose from data set.
mean_pose = np.load('./mean_pose.npy')

# get max val from dataset
max_val = np.load('./data_max.npy')

# where to save the saved network weights after training
PATH = './MoGapSaveState.pth'

# path to data that has been cropped into smaller same-sized chucks
cropped_ground_truth_path_train = '/run/media/henryp/HenryHDD/DataSets/CMU/Formatted/Train/Cropped_data/sliced_'

# cropped_ground_truth_path_test = '/run/media/henryp/HenryHDD/DataSets/CMU/Formatted/Test/Cropped_data'

if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('============')
print('Training on: ' + str(device))
print('============')

training_partition, training_set_size = load_data(ground_truth_dir=cropped_ground_truth_path_train,
                                                  type='train')


training_set = DataSet(list_IDS=training_partition['train'],
                       data_dir=cropped_ground_truth_path_train,
                       clip_length=64)
training_generator = data.DataLoader(training_set, **params)

# testing_partition, testing_set_size = load_data(ground_truth_dir=cropped_ground_truth_path_test,
#                                                 type='test')
# testing_set = DataSet(list_IDS=testing_partition['test'],
#                       data_dir=cropped_ground_truth_path_test,
#                       clip_length=64)
#
# testing_generator = data.DataLoader(testing_set, **params)

# load the network class
# input size is the number of features in your input data
net = LSTMAE(input_size=123, enc_first_size=62, enc_second_size=31,
             dec_first_size=31, output_size=123, num_layers=1)
net = net.cuda()

if train_network:

    # select network hyper parameters
    criterion = RMSE_loss()
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=5, verbose=True)
    # track loss per epoch to plot how the network learns over the number of epochs
    loss_values = []

    # training loop
    for epoch in range(max_epochs):
        for local_batch in training_generator:

            # shuffle_tensor_order
            permute = torch.randperm(local_batch.size(0))
            local_batch = local_batch[permute]

            # apply missing markers
            local_batch_missing = apply_missing_cmu_val(time_series_tensor=local_batch, erasures_perc=10)

            # move to GPU
            local_batch = local_batch.to(device)
            local_batch_missing = local_batch_missing.to(device)

            # training loop
            optimizer.zero_grad()
            outputs = net(local_batch_missing)

            # backwards step, RMSE loss
            loss = criterion(outputs, local_batch)
            loss.backward()
            optimizer.step()
        scheduler.step(loss)

        print('epoch [{}/{}], RMSE_loss:{:.8f}'
              .format(epoch + 1, max_epochs, loss.data))
        loss_values.append(loss.data.item())

    torch.save(net.state_dict(), PATH)

    # plot loss over training
    plt.plot([i for i in range(max_epochs)], loss_values)
    plt.title(architecture + ' loss || Final RMSEloss: ' + str(loss_values[-1]))
    plt.savefig(architecture + '_loss')

#
# else:
#     print('Loading saved network state...')
#     net.load_state_dict(torch.load(PATH))

# # test on testing_set
# criterion = nn.MSELoss()
# with torch.no_grad():
#     for local_batch in testing_generator:
#         # preprocess data
#         local_batch = local_batch.float()
#         local_batch_missing = apply_missing_cmu_val(time_series_tensor=local_batch, erasures_perc=10)
#
#         # move to GPU
#         local_batch = local_batch.to(device)
#         local_batch_missing = local_batch_missing.to(device)
#
#         outputs = net(local_batch_missing)
#         loss = criterion(local_batch, outputs)
#
#     print('Testing RMSELoss:{:.8f}'.format(torch.sqrt(loss.data)))
