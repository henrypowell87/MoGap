import torch.nn as nn


class EncoderCNNLSTM(nn.Module):
    def __init__(self, num_frames, input_size, enc_first_size, enc_second_size, num_layers):
        super(EncoderCNNLSTM, self).__init__()
        self.num_frames = num_frames
        self.input_size = input_size
        self.enc_first_size = enc_first_size
        self.enc_second_size = enc_second_size
        self.num_layers = num_layers

        self.cnn_one = nn.Conv1d(in_channels=num_frames,
                                 out_channels=num_frames,
                                 kernel_size=10)
        self.cnn_two = nn.Conv1d(in_channels=num_frames,
                                 out_channels=num_frames,
                                 kernel_size=5)
        self.relu = nn.ReLU(inplace=True)

        self.LSTM_enc_one = nn.LSTM(input_size=17,
                                    hidden_size=enc_first_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        self.LSTM_enc_two = nn.LSTM(input_size=enc_first_size,
                                    hidden_size=enc_second_size,
                                    num_layers=num_layers,
                                    batch_first=True)

    def forward(self, x):
        x = self.cnn_one(x)
        x = self.relu(x)
        x = self.cnn_two(x)
        x = self.relu(x)
        x, (hidden, cell) = self.LSTM_enc_one(x, None)
        x, (hidden, cell) = self.LSTM_enc_two(x, None)
        return x


class DecoderCNNLSTM(nn.Module):
    def __init__(self, enc_second_size, dec_first_size, output_size, num_layers):
        super(DecoderCNNLSTM, self).__init__()
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
        x, (hidden, cell) = self.LSTM_dec_one(x, None)
        x, (hidden, cell) = self.LSTM_dec_two(x, None)
        return x


class CNNLSTMAE(nn.Module):
    def __init__(self, num_frames, input_size,
                 enc_first_size, enc_second_size,
                 dec_first_size, output_size, num_layers):
        super(CNNLSTMAE, self).__init__()
        self.encoder = EncoderCNNLSTM(num_frames, input_size, enc_first_size, enc_second_size, num_layers)
        self.decoder = DecoderCNNLSTM(enc_second_size, dec_first_size, output_size, num_layers)

    def forward(self, x):
        encoded_input = self.encoder(x)
        decoded_output = self.decoder(encoded_input)
        return decoded_output