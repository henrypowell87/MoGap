import torch.nn as nn


class EncoderBdLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderBdLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.LSTM_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                                bidirectional=True)

    def forward(self, x):
        pred, hidden = self.LSTM_enc(x, None)
        return pred


class DecoderBdLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderBdLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.LSTM_dec = nn.LSTM(input_size=hidden_size*2, hidden_size=output_size // 2, num_layers=num_layers,
                                batch_first=True, bidirectional=True)

    def forward(self, x):
        pred, hidden = self.LSTM_dec(x, None)
        return pred


class BdLSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BdLSTMAE, self).__init__()
        self.encoder = EncoderBdLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderBdLSTM(hidden_size, input_size, num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
