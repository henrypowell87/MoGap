import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # convolutional layers
        self.CNN = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.CNN_b = nn.Conv1d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.Dropout = nn.Dropout(0.5)
        self.Max_Pooling = nn.MaxPool1d(kernel_size=3, stride=2)

        # LSTM layer
        self.LSTM_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = self.CNN(x)
        x = self.CNN_b(x)
        x = self.Dropout(x)
        x = self.Max_Pooling(x)
        pred, hidden = self.LSTM_enc(x, None)
        return pred


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.LSTM_dec = nn.LSTM(input_size=hidden_size, hidden_size=output_size, num_layers=num_layers,
                                batch_first=True)

    def forward(self, x):
        pred, hidden = self.LSTM_dec(x, None)
        return pred


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(hidden_size, input_size, num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output