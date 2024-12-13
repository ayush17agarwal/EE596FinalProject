import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch

def get_data(path):
  cwd = os.getcwd()

  amzn = pd.read_csv(path)
  amzn_np = amzn.to_numpy()

  amzn_no_splits = []

  for i in amzn_np:
    date = i[0]
    # Our dataset is off by exactly 1 years and 11 months.
    # AMZN stock split occurred on 06/06/2022
    if date < '2020-07-06' and date >= '1997-12-01':
      amzn_no_splits.append(i)

  no_split_np = np.array(amzn_no_splits)

  close_data = no_split_np[:, -3]
  stock_dates = no_split_np[:, 0]
  
  close_data = np.flip(close_data)
  stock_dates = np.flip(stock_dates)

  return amzn_np, close_data, stock_dates

def scale_data(data):
  ms = MinMaxScaler()

  return ms.fit_transform(data.reshape(len(data),1))


def simple_plot(data, xlabel, ylabel, title, plot_save_path = None):
  plt.figure(figsize=(10,5))
  plt.plot(data)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  
  if plot_save_path:
    plt.savefig(plot_save_path)

  plt.show()

def simple_plot(data1, data2, xlabel, ylabel, title, plot_save_path = None):
  plt.figure(figsize=(10,5))
  plt.plot(data1)
  plt.plot(data2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  
  if plot_save_path:
    plt.savefig(plot_save_path)

  plt.show()

def predict_stock_price(truth_seq, model, test_size, bias, amplitude, num_features=1, encoder_inputseq_len=7, decoder_outputseq_len=1):

  #print(y_test)

  # initialize empty torch tensor array to store decoder output sequence
  # This should be the same size as the test sequence
  decoder_output_seq = torch.zeros(test_size, num_features)

  # First n-datapoints in decoder output sequence = First n-datapoints in ground truth test sequence
  # n = encoder_input_seq_len
  decoder_output_seq[:encoder_inputseq_len] = truth_seq[:encoder_inputseq_len]

  # Initialize index for prediction
  pred_start_ind = 0

  # Activate no_grad() since we aren't performing backprop
  with torch.no_grad():

      # Loop continues until the RNN prediction reaches the end of the testing sequence length
      while pred_start_ind + encoder_inputseq_len + decoder_outputseq_len < test_size:

          # initialize hidden state for encoder
          hidden_state = None

          # Define the input to encoder
          input_test_seq = truth_seq[pred_start_ind:pred_start_ind + encoder_inputseq_len]
          # Add dimension to first dimension to keep the input (sample_size, seq_len, # of features/timestep)
          input_test_seq = torch.unsqueeze(input_test_seq, 0)

          # Feed the input to encoder and set resulting hidden states as input hidden states to decoder
          encoder_output, encoder_hidden = model.Encoder(input_test_seq, hidden_state)
          decoder_hidden = encoder_hidden

          # Initial input to decoder is last timestep feature from the encoder input sequence
          decoder_input = input_test_seq[:, -1, :]
          # Add dimension to keep the input (sample_size, seq_len, # of features/timestep)
          decoder_input = torch.unsqueeze(decoder_input, 2)

          # Populate decoder output sequence
          for t in range(decoder_outputseq_len):

              # Generate new output for timestep t
              decoder_output, decoder_hidden = model.Decoder(decoder_input, decoder_hidden)
              # Populate the corresponding timestep in decoder output sequence
              decoder_output_seq[pred_start_ind + encoder_inputseq_len + t] = (torch.squeeze(decoder_output) * amplitude) - bias
              # Use the output of the decoder as new input for the next timestep
              decoder_input = decoder_output

          # Update pred_start_ind
          pred_start_ind += decoder_outputseq_len

      return decoder_output_seq
