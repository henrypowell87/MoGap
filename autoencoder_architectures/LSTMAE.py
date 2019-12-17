import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
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