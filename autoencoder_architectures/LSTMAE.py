import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, enc_first_size, enc_second_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.enc_first_size = enc_first_size
        self.enc_second_size = enc_second_size
        self.num_layers = num_layers

        self.LSTM_enc_one = nn.LSTM(input_size=input_size,
                                    hidden_size=enc_first_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        self.LSTM_enc_two = nn.LSTM(input_size=enc_first_size,
                                    hidden_size=enc_second_size,
                                    num_layers=num_layers,
                                    batch_first=True)

    def forward(self, x):
        pred, (hidden, cell) = self.LSTM_enc_one(x, None)
        pred, (hidden, cell) = self.LSTM_enc_two(pred, None)
        return pred


class DecoderLSTM(nn.Module):
    def __init__(self, enc_second_size, dec_first_size, output_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.enc_second_size = enc_second_size
        self.dec_first_size = dec_first_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.LSTM_dec_one = nn.LSTM(input_size=enc_second_size,
                                    hidden_size=dec_first_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        self.LSTM_dec_two = nn.LSTM(input_size=dec_first_size,
                                    hidden_size=output_size,
                                    num_layers=num_layers,
                                    batch_first=True)

    def forward(self, x):
        pred, (hidden, cell) = self.LSTM_dec_one(x, None)
        pred, (hidden, cell) = self.LSTM_dec_two(pred, None)
        return pred


class LSTMAE(nn.Module):
    def __init__(self, input_size, enc_first_size, enc_second_size, dec_first_size, output_size, num_layers):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderLSTM(input_size, enc_first_size, enc_second_size, num_layers)
        self.decoder = DecoderLSTM(enc_second_size, dec_first_size, output_size, num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output