import numpy as np
import torch

from utils import *
from model import Encoder_Decoder

def generate_input_output_seqs(y, encoder_inputseq_len, decoder_outputseq_len, stride = 1, num_features = 1):

    L = y.shape[0] # Length of y

    # Calculate how many input/target sequences there will be based on the parameters and stride
    num_samples = (L - encoder_inputseq_len - decoder_outputseq_len) // stride + 1

    # Numpy zeros arrray to contain the input/target sequences
    # Note that they should be in (num_samples, seq_len, num_features/time step) format
    train_input_seqs = np.zeros([num_samples, encoder_inputseq_len, num_features])
    train_output_seqs = np.zeros([num_samples, decoder_outputseq_len, num_features])

    # Iteratively fill in train_input_seqs and train_output_seqs
    # See slide 17 of lab 7 to get an idea of how input_seqs and output_seqs look like
    for ff in np.arange(num_features):

        for ii in np.arange(num_samples):

            start_x = stride * ii
            end_x = start_x + encoder_inputseq_len
            train_input_seqs[ii, :, ff] = y[start_x:end_x, ff]

            start_y = stride * ii + encoder_inputseq_len
            end_y = start_y + decoder_outputseq_len
            train_output_seqs[ii, :, ff] = y[start_y:end_y, ff]

    return train_input_seqs, train_output_seqs

def get_batches(data, batchsize, encoder_inputseq_len = 7, decoder_outputseq_len = 1, ):
  train_input_seqs, train_output_seqs = generate_input_output_seqs(y = data,
                                                                 encoder_inputseq_len = encoder_inputseq_len,
                                                                 decoder_outputseq_len = decoder_outputseq_len,
                                                                 stride = 1,
                                                                 num_features = 1)

  # Check the dimensions of encoder input seqs and decoder output seqs
  print("Encoder Training Inputs Shape: ", train_input_seqs.shape)
  print("Decoder Training Outputs Shape: ", train_output_seqs.shape)

  # Convert training dataset into torch tensors
  training_input_seqs = torch.from_numpy(train_input_seqs).float()
  training_output_seqs = torch.from_numpy(train_output_seqs).float()

  # Split the training dataset to mini-batches
  # Skipping the last mini-batch since its size can be smaller than the set batchsize
  train_batches_features = torch.split(training_input_seqs, batchsize)[:-1]
  train_batches_targets = torch.split(training_output_seqs, batchsize)[:-1]

  # Total number of mini-batches in the training set
  batch_split_num = len(train_batches_features)

  return train_batches_features, train_batches_targets, batch_split_num

def train_LSTM(epochs, batch_split_num, train_batches_features, train_batches_targets, model, loss_func, optimizer, num_features, batchsize):
  train_loss_list = []
  for epoch in range(epochs): # For each epoch

      for k in range(batch_split_num): # For each mini_batch

          # initialize hidden states to Encoder
          hidden_state = None

          # initialize empty torch tensor array to store decoder output sequence
          decoder_output_seq = torch.zeros(batchsize, decoder_outputseq_len, num_features)

          # empty gradient buffer
          optimizer.zero_grad()

          # Feed k-th mini-batch for encoder input sequences to encoder with hidden state
          encoder_output, encoder_hidden = model.Encoder(train_batches_features[k], hidden_state)
          # Re-define the resulting encoder hidden states as input hidden states to decoder
          decoder_hidden = encoder_hidden

          # Initial input to decoder is last timestep feature from the encoder input sequence
          decoder_input = train_batches_features[k][:, -1, :]
          # The extracted feature is 2D so need to add additional 3rd dimension
          # to conform to (sample size, seq_len, # of features)
          decoder_input = torch.unsqueeze(decoder_input, 2)

          # Populating the decoder output sequence
          for t in range(decoder_outputseq_len): # for each timestep in output sequence

              # Feed in the decoder_input and decoder_hidden to Decoder, get new output and hidden states
              decoder_output, decoder_hidden = model.Decoder(decoder_input, decoder_hidden)

              # Populate the corresponding timestep in decoder output sequence
              decoder_output_seq[:, t, :] = torch.squeeze(decoder_output, 2)

              # We are using teacher forcing so using the groundtruth training target as the next input
              decoder_input = train_batches_targets[k][:, t, :]

              # The extracted feature is 2D so need to add additional 3rd dimension
              # to conform to (sample size, seq_len, # of features)
              decoder_input = torch.unsqueeze(decoder_input, 2)

          # Compare the predicted decoder output sequence aginast the target sequence to compute the MSE loss
          loss = loss_func(torch.squeeze(decoder_output_seq), torch.squeeze(train_batches_targets[k]))

          # Save the loss
          train_loss_list.append(loss.item())

          # Backprop
          loss.backward()

          # Update the RNN
          optimizer.step()

      print("Averaged Training Loss for Epoch ", epoch,": ", np.mean(train_loss_list[-batch_split_num:]))

  return train_loss_list

if __name__ == '__main__':
  print("hello")

  data_path = '../data/AMZN.csv'

  _, closing_price, dates = get_data(data_path, '2020-07-06', '1997-12-01')

  closing_price = scale_data(closing_price)

  simple_plot(closing_price, "days", "stock price", "AMZN stock price from 01/01/2000 to 06/06/2022", "../results/graphs/TrainingClosingPrice.png")

  num_datapoints = closing_price.size
  print("Length of the dataset:", num_datapoints, "days")

  encoder_inputseq_len = 7 # num days
  decoder_outputseq_len = 1 # num outputs (predicting next day)
  testing_sequence_len = 7 # num days

  test_size = int(0.5 * num_datapoints) # 50% of our data is for testing sequence

  y_train = closing_price[test_size:] # arr[start_pos:end_pos:skip]
  y_test = closing_price[:test_size]

  y_train_dates = dates[test_size:]
  y_test_dates = dates[:test_size]

  ################################
  ######  Training model 1  ######
  ################################

  # Fix random seed
  torch.manual_seed(42)

  # Using input_size = 1 (# of features to be fed to RNN per timestep)
  # Using decoder_output_size = 1 (# of features to be output by Decoder RNN per timestep)
  Encoder_Decoder_RNN = Encoder_Decoder(input_size = 1, hidden_size = 15,
                                        decoder_output_size = 1, num_layers = 1)

  # Define learning rate + epochs
  learning_rate = 0.0005
  epochs = 20

  # Define batch size and num_features/timestep (this is simply the last dimension of train_output_seqs)
  batchsize = 5
  num_features = 1 # train_output_seqs.shape[2]

  # Define loss function/optimizer
  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(Encoder_Decoder_RNN.parameters(), lr=learning_rate)

  # print(Encoder_Decoder_RNN)
  torch.save(Encoder_Decoder_RNN, "../checkpoints/Encoder_Decoder_RNN.pt")

  train_batches_features, train_batches_targets, batch_split_num = get_batches(y_train.reshape((-1, 1)), batchsize = batchsize)

  # - bias
  train_loss_list = train_LSTM(
     epochs, 
     batch_split_num, 
     train_batches_features, 
     train_batches_targets, 
     Encoder_Decoder_RNN, 
     loss_func, 
     optimizer, 
     num_features, 
     batchsize)
  

  simple_plot(np.convolve(train_loss_list, np.ones(100), 'valid') / 100, "training loss", "Iterations", "Training Loss for Predicting Stock Price", "../results/graphs/PredictedStockPriceLSTMTrainingLoss.png")

  ################################
  ######  Testing model 1  #######
  ################################

  test_input_seq = torch.from_numpy(np.array(y_test,dtype="float64")).float()
  test_input_seq= test_input_seq.reshape((-1,1))

  decoder_output_seq = predict_stock_price(test_input_seq, Encoder_Decoder_RNN, test_size, .036, 1)

  simple_plot(test_input_seq, decoder_output_seq, "days", "stock price (normalized)", "RNN Predicted vs GroundTruth for Predicting Stock Price", "../results/graphs/LSTMPredictedVSGroundTruth.png")

  #######################################
  ######  Model 2 Pre-processing  #######
  #######################################


  test_size_residuals = int(0.5 * y_test.size) # 50% of our data is for testing sequence

  residuals_train_dates = y_test_dates[test_size_residuals:]
  residuals_test_dates = y_test_dates[:test_size_residuals]

  y_2_train = y_test[test_size_residuals:] # arr[start_pos:end_pos:skip]

  y_2_test = y_test[:test_size_residuals]

  decoder_output_train = decoder_output_seq[test_size_residuals:]

  decoder_output_test = decoder_output_seq[:test_size_residuals]

  #Residuals: Difference between the predicted stock price and the actual stock price

  residuals_train = torch.from_numpy(np.array(y_2_train,dtype="float64")).float() - decoder_output_train

  residuals_test = torch.from_numpy(np.array(y_2_test,dtype="float64")).float() - decoder_output_test

  print(residuals_train.shape)

  print(residuals_test.shape)

  ################################ 
  ######  Training model 2  ######
  ################################

  # Fix random seed
  torch.manual_seed(42)

  # Using input_size = 1 (# of features to be fed to RNN per timestep)
  # Using decoder_output_size = 1 (# of features to be output by Decoder RNN per timestep)
  Encoder_Decoder_RNN_Residual = Encoder_Decoder(input_size = 1, hidden_size = 15,
                                        decoder_output_size = 1, num_layers = 1)

  # Define learning rate + epochs
  learning_rate = 0.0005
  epochs = 10

  # Define batch size and num_features/timestep (this is simply the last dimension of train_output_seqs)
  batchsize = 5
  num_features = 1 # train_output_seqs.shape[2]

  # Define loss function/optimizer
  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(Encoder_Decoder_RNN_Residual.parameters(), lr=learning_rate)

  # print(Encoder_Decoder_RNN_Residual)
  torch.save(Encoder_Decoder_RNN_Residual, "../checkpoints/Encoder_Decoder_RNN_Residual.pt")

  train_batches_features, train_batches_targets, batch_split_num = get_batches(residuals_train.reshape((-1, 1)), batchsize = batchsize)

  train_loss_list = train_LSTM(
     epochs, 
     batch_split_num, 
     train_batches_features, 
     train_batches_targets, 
     Encoder_Decoder_RNN_Residual, 
     loss_func, optimizer, 
     num_features, 
     batchsize)
  
  simple_plot(np.convolve(train_loss_list, np.ones(100), 'valid') / 100, "training loss", "Iterations", "Training Loss for Residuals", "../results/graphs/ResidualsLSTMTrainingLoss.png")

  ################################
  ######  Testing model 2  #######
  ################################

  #Use the last 25% as data to detect anomalies
  test_input_seq = torch.from_numpy(np.array(residuals_test,dtype="float64")).float()

  print(test_input_seq.shape)
  test_input_seq= test_input_seq.reshape((-1,1))

  print(test_input_seq.shape)


  decoder_output_seq = predict_stock_price(test_input_seq, Encoder_Decoder_RNN_Residual, test_size_residuals, .0485, -80)

  simple_plot(test_input_seq, decoder_output_seq, "days", "residuals (normalized)", "RNN Predicted Residuals vs GroundTruth", "../results/graphs/LSTMPredictedResidualsVSGroundTruth.png")

  ##################################
  ######  Anomaly Detection  #######
  ##################################

  #Predicted residuals - actual residuals
  residual_differences = decoder_output_seq - test_input_seq

  # print(residual_differences)
  # print(residuals_test_dates)

  print(residuals_test_dates.shape)
  print(residual_differences.shape)
  print(torch.max(residual_differences))
  print(torch.min(residual_differences))

  absolute_residual_differences = torch.abs(residual_differences)

  condition = (absolute_residual_differences > .007)

  filtered_tensor = torch.where(condition, 1, 0)

  print(filtered_tensor.shape)

  filtered_array = filtered_tensor.numpy()


  anomaly_dates = []


  for i in range(len(filtered_array)):
    if filtered_array[i] == 1:
      anomaly_dates.append(residuals_test_dates[i])


  print(anomaly_dates)
