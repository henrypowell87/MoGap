import torch.nn as nn


class EncoderBdRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderBdRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_enc = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

    def forward(self, x):
        pred, hidden = self.rnn_enc(x, None)
        return pred


class DecoderBdRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderBdRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.rnn_dec = nn.RNN(input_size=hidden_size * 2, hidden_size=output_size // 2, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

    def forward(self, x):
        pred, hidden = self.rnn_dec(x, None)
        return pred


class BdRNNAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BdRNNAE, self).__init__()
        self.encoder = EncoderBdRNN(input_size, hidden_size, num_layers)
        self.decoder = DecoderBdRNN(hidden_size, input_size, num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output