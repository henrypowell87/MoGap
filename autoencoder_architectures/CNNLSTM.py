import torch.nn as nn
from torch import flatten


class EncoderCNNLSTM(nn.Module):
    def __init__(self, num_frames):
        super(EncoderCNNLSTM, self).__init__()
        self.num_frames = num_frames

        # Convolutional layer with maxpooling and ReLU activation
        self.cnn_one = nn.Conv1d(in_channels=num_frames,
                                 out_channels=128,
                                 kernel_size=10)

        self.cnn_two = nn.Conv1d(in_channels=128,
                                 out_channels=256,
                                 kernel_size=10)

        self.relu = nn.ReLU(inplace=True)

        self.max_pool_one = nn.MaxPool1d(kernel_size=4, stride=1)


    def forward(self, x):
        x = self.cnn_one(x)
        x = self.relu(x)
        x = self.cnn_two(x)
        x = self.relu(x)
        x = self.max_pool_one(x)
        return x


class DecoderCNNLSTM(nn.Module):
    def __init__(self, num_layers):
        super(DecoderCNNLSTM, self).__init__()
        self.num_layers = num_layers

        self.LSTM_dec_one = nn.LSTM(input_size=9,
                                    hidden_size=15,
                                    num_layers=num_layers,
                                    batch_first=True)

        self.LSTM_dec_two = nn.LSTM(input_size=15,
                                    hidden_size=20,
                                    num_layers=num_layers,
                                    batch_first=True)

        self.dcnn = nn.ConvTranspose1d(in_channels=256, out_channels=64, kernel_size=11)

    def forward(self, x):
        x, (hidden, cell) = self.LSTM_dec_one(x, None)
        x, (hidden, cell) = self.LSTM_dec_two(x, None)
        x = self.dcnn(x)
        return x


class CNNLSTMAE(nn.Module):
    def __init__(self, num_frames, num_layers):
        super(CNNLSTMAE, self).__init__()
        self.encoder = EncoderCNNLSTM(num_frames=num_frames)
        self.decoder = DecoderCNNLSTM(num_layers=num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
