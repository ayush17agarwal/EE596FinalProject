import torch

class Encoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):

        super(Encoder, self).__init__()

        # Using LSTM for Encoder with batch_first = True

        self.lstm = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                                  num_layers = num_layers,
                                  batch_first = True)

    def forward(self, input_seq, hidden_state):

        lstm_out, hidden = self.lstm(input_seq, hidden_state)

        return lstm_out, hidden

class Decoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):

        super(Decoder, self).__init__()

        # Using LSTM for Decoder with batch_first = True
        # fc_decoder for converting hidden states -> single number

        self.lstm = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                                  num_layers = num_layers,
                                  batch_first = True)

        self.fc_decoder = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, encoder_hidden_states):

        lstm_out, hidden = self.lstm(input_seq, encoder_hidden_states)
        output = self.fc_decoder(lstm_out)

        return output, hidden

class Encoder_Decoder(torch.nn.Module):

    def __init__(self, input_size, hidden_size, decoder_output_size, num_layers):

        # Combine Encoder and Decoder classes into one

        super(Encoder_Decoder, self).__init__()

        self.Encoder = Encoder(input_size = input_size, hidden_size = hidden_size,
                               num_layers = num_layers)

        self.Decoder = Decoder(input_size = input_size, hidden_size = hidden_size,
                               output_size = decoder_output_size, num_layers = num_layers)