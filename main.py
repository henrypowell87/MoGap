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

PATH = './MoGapSaveState.pth'

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

if train_network:

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=1e-6, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=5, verbose=True)

    for epoch in range(max_epochs):
        for local_batch in training_generator:

            if epoch % 10 == 0:
                local_batch = local_batch.clone()
                torch.save(local_batch, './examples/' + 'original_epoch' + str(epoch + 1) + '.pt')

            # preprocess data
            local_batch = normalize_time_series(local_batch, -1, 1)
            local_batch_missing = apply_missing(time_series_tensor=local_batch, max_erasures=5, max_gap_size=10)

            # move to GPU
            local_batch = local_batch.to(device)
            local_batch_missing = local_batch_missing.to(device)

            if epoch % 10 == 0:
                missing = local_batch_missing.clone()
                torch.save(missing, './examples/' + 'missing_epoch' + str(epoch+1) + '.pt')

            # training loop
            outputs = net(local_batch_missing)
            if epoch % 10 == 0:
                estimated = outputs.clone()
                torch.save(estimated, './examples/' + 'estimated_epoch' + str(epoch+1) + '.pt')

            loss = criterion(outputs, local_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(loss)

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, max_epochs, loss.data.item()))

    torch.save(net.state_dict(), PATH)

else:
    print('Loading saved network state...')
    net.load_state_dict(torch.load(PATH))

# number of samples you want to run through the network
num_samples = 1

# keep track of how many samples have been processed
processed = 0

# run the required number of samples through the trained network
with torch.no_grad():
    i = 1
    for local_batch in training_generator:
        original = local_batch[0].unsqueeze(0)
        original = normalize_time_series(original, -1, 1)
        original_missing = apply_missing(time_series_tensor=original, max_erasures=5, max_gap_size=10)

        original = original.to(device)
        original_missing = original_missing.to(device)

        estimated = net(original)

        torch.save(original, './examples/original_' + str(i) + '.pt')
        torch.save(original_missing, './examples/original_missing_' + str(i) + '.pt')
        torch.save(estimated, './examples/estimated_' + str(i) + '.pt')

        processed += 1
        i += 1
        if processed == num_samples:
            break




