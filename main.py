"""
Author: Henry Powell
Institution: Institute of Neuroscience and Psychology, Glasgow University, Scotland.

High level script to train a denoising autoencoder on a given data set. Different autoencoder models
can be imported from the 'autoencoder_architectures' directory.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from data_loader import DataSet
from RMSE_loss import RMSE_loss
from autoencoder_architectures.CNNLSTM import CNNLSTMAE
from functions import load_data, apply_missing_cmu_val, crop_to_missing, denormalize_series


# set script params here
architecture = 'CNNLSTMAE'
# path_to_ground_truth = '/home/henryp/PycharmProjects/MoGap/filtered_tensors'
batch_size = 17
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 90
# pick whether you want to train the network on GPU or CPU
run_on = 'GPU'
# change to false if you want to preload the network with saved weights
train_network = True

# get mean pose from data set.
mean_pose = torch.as_tensor(np.load('./mean_pose.npy')).cuda()

# get max val from dataset
max_val = torch.as_tensor(np.load('./data_max.npy')).cuda()

# where to save the saved network weights after training
PATH = './MoGapSaveState' + architecture + '_test.pth'

# path to data that has been cropped into smaller same-sized chucks
cropped_ground_truth_path_train = '/run/media/henryp/HenryHDD/DataSets/CMU/Formatted/Train/Cropped_data/sliced_'

cropped_ground_truth_path_test = '/run/media/henryp/HenryHDD/DataSets/CMU/Formatted/Test/Cropped_data/'

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

testing_partition, testing_set_size = load_data(ground_truth_dir=cropped_ground_truth_path_test,
                                                type='test')

testing_set = DataSet(list_IDS=testing_partition['test'],
                      data_dir=cropped_ground_truth_path_test,
                      clip_length=64)

testing_generator = data.DataLoader(testing_set, **params)

# load the network class
# input size is the number of features in your input data
net = CNNLSTMAE(num_frames=64, num_layers=2)
net = net.cuda()

if train_network:

    # select network hyper parameters
    criterion = RMSE_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                                        patience=5, verbose=True)

    # track loss per epoch to plot how the network learns over the number of epochs
    loss_values = []
    best_loss = 1000

    # training loop
    for epoch in range(max_epochs):
        epoch_acc = torch.Tensor().cuda()
        for local_batch in training_generator:

            # apply missing markers and split missing and ground truth data
            local_batch = local_batch.float()
            local_batch_missing, _ = apply_missing_cmu_val(time_series_tensor=local_batch,
                                                           erasures_perc=10,
                                                           missing_val=1.0001)
            # move to GPU
            local_batch = local_batch.to(device)
            local_batch_missing = local_batch_missing.to(device)

            # training loop
            outputs = net(local_batch_missing)

            # get only missing data
            y, y_hat = crop_to_missing(local_batch, local_batch_missing, outputs, nan_val=1.0001)

            # loss = criterion(outputs, local_batch)
            loss = criterion(y, y_hat)
            epoch_acc = torch.cat((epoch_acc, loss.data.unsqueeze(0)))
            # backwards step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 10)
            optimizer.step()
        # scheduler.step(loss)
        print('epoch [{}/{}], mean RMSE_loss for epoch:{:.8f}'
              .format(epoch + 1, max_epochs, torch.mean(epoch_acc)))

        loss_values.append(torch.Tensor.cpu(torch.mean(epoch_acc)).numpy())

        # save best epoch
        if loss.data < best_loss:
            best_loss = loss.data
            torch.save(net.state_dict(), PATH)

    # plot loss over training
    plt.plot([i for i in range(max_epochs)], loss_values)
    plt.title(architecture + ' loss || Final RMSEloss: ' + str(loss_values[-1]))
    plt.savefig(architecture + '_loss')


else:
    print('Loading saved network state...')
    net.load_state_dict(torch.load(PATH))

# test on testing_set
criterion = RMSE_loss()
testing_loss = []
with torch.no_grad():
    for local_batch in testing_generator:
        # preprocess data
        local_batch = local_batch.float()
        local_batch_missing, _ = apply_missing_cmu_val(time_series_tensor=local_batch,
                                                       erasures_perc=10,
                                                       missing_val=1.0001)

        # move to GPU
        local_batch = local_batch.to(device)
        local_batch_missing = local_batch_missing.to(device)

        outputs = net(local_batch_missing)
        loss = criterion(local_batch, outputs)
        testing_loss.append(torch.Tensor.cpu(loss).data)

    print('Mean RMSE_loss for testing set:{:.8f}'.format(torch.mean(torch.as_tensor(testing_loss))))
