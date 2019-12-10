import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from torch.utils import data

# add ground_truths
class DataSet(data.Dataset):
    def __init__(self, list_IDS, data_dir, transform=None):
        # self.ground_truths = ground_truths
        self.list_IDS = list_IDS
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_IDS)

    def __getitem__(self, index):
        ID = self.list_IDS[index]
        x = np.load(self.data_dir + ID)
        if self.transform:
            x = self.transform(x)
        return x


class RNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        super(RNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.RNN(input_size=hidden_size, hidden_size=input_size, num_layers=num_layers, batch_first=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, hidden_a = self.encoder(x)
        x, hidden_b = self.decoder(x)
        return x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_enc = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        pred, hidden = self.rnn_enc(x, None)
        return pred


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn_dec = nn.RNN(input_size=hidden_size, hidden_size=output_size, num_layers=num_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pred, hidden = self.rnn_dec(x, None)
        pred = self.sigmoid(pred)
        return pred


class RNNAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output