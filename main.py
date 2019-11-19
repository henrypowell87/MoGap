import torch.nn
from torch.utils import data
from classes import DataSet
from functions import load_data

batch_size = 10
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 90
run_on = 'GPU'
train_network = True

gap_data_path = '/home/henryp/PycharmProjects/MoGap/augmented_data/'
ground_truth_path = '/home/henryp/PycharmProjects/MoGap/ground_truth_data/'

if run_on == 'GPU':
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Training on: ' + str(device))

partition, ground_truth, training_set_size = load_data(gap_data_dir=gap_data_path, ground_truth_dir=ground_truth_path)

training_set = DataSet(partition['train'], ground_truth, data_dir=gap_data_path)
training_generator = data.DataLoader(training_set, **params)

for local_batch, local_ground_truth in training_generator:
    print(local_batch[0].size())
    print(local_ground_truth[0].size())
