import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_a_enc = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.rnn_b_enc = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.relu_enc = nn.ReLU()

    def forward(self, x):
        x, hidden_a = self.rnn_a_enc(x, None)
        # x, hidden_b = self.rnn_b_enc(x, None)
        # x = self.relu_enc(x)
        return x


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # self.rnn_a_dec = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.rnn_b_dec = nn.RNN(input_size=hidden_size, hidden_size=output_size, num_layers=num_layers, batch_first=True)
        self.relu_dec = nn.ReLU()

    def forward(self, x):
        # x, hidden_a = self.rnn_a_dec(x, None)
        x, hidden_b = self.rnn_b_dec(x, None)
        # x = self.relu_dec(x)
        return x


class RNNAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output