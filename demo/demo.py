
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

from utils import *

if __name__ == '__main__':

  data_path = '../data/AMZN.csv' 
  amzn_np, _, _ = get_data(data_path, '2020-07-06', '1997-12-01')

  #Post stock split calculation
  amzn_post_split = []
  y_test_dates = []
  for i in amzn_np:
    date = i[0]
    # Our dataset is off by exactly 1 years and 11 months.
    # AMZN stock split occurred on 06/06/2022
    if date > '2020-08-06':
      amzn_post_split.append(i)
      y_test_dates.append(date)

  amzn_post_split = np.array(amzn_post_split)
  #get only the closing prices for the stock

  amzn_post_split = amzn_post_split[:, -3]
  amzn_post_split = np.flip(amzn_post_split)

  y_test_dates = np.flip(y_test_dates)
  Ms = MinMaxScaler()
  scaled_amzn_post_split = Ms.fit_transform(amzn_post_split.reshape(len(amzn_post_split),1))

  simple_plot(scaled_amzn_post_split, "days", "stock price", "AMZN stock price from 06/06/2022 to now", "../result/graphs/AMZNStockPrice_Post_6_6_2022.png")

  test_input_seq = torch.from_numpy(np.array(scaled_amzn_post_split,dtype="float64")).float()

  print(test_input_seq.shape)
  test_input_seq= test_input_seq.reshape((-1,1))

  print(test_input_seq.shape)
  test_size = test_input_seq.shape[0]
  print(test_size)

  Encoder_Decoder_RNN = torch.load("../checkpoints/Encoder_Decoder_RNN.pt", weights_only=False)
  Encoder_Decoder_RNN.eval()

  decoder_output_seq = predict_stock_price(test_input_seq, Encoder_Decoder_RNN, test_size, .036, 1)
  simple_plot(test_input_seq, decoder_output_seq, "days", "stock price (normalized)", "RNN Predicted vs GroundTruth for Predicting Stock Price", "../result/graphs/ValidationDataPredictingStockPrice.png")

  #Calculate residuals

  residuals_test_dates = y_test_dates

  #Residuals: Difference between the predicted stock price and the actual stock price
  residuals_test = torch.from_numpy(np.array(test_input_seq,dtype="float64")).float() - decoder_output_seq

  print(residuals_test.shape)

  test_input_seq = torch.from_numpy(np.array(residuals_test,dtype="float64")).float()

  print(test_input_seq.shape)
  test_input_seq= test_input_seq.reshape((-1,1))

  print(test_input_seq.shape)

  Encoder_Decoder_RNN_Residual = torch.load("../checkpoints/Encoder_Decoder_RNN_Residual.pt", weights_only=False)
  Encoder_Decoder_RNN_Residual.eval()

  decoder_output_seq = predict_stock_price(test_input_seq, Encoder_Decoder_RNN_Residual, test_size, .0485, -80)

  simple_plot(test_input_seq, decoder_output_seq, "days", "residuals(normalized)", "RNN Predicted vs GroundTruth for Predicting Stock Price", "../result/graphs/ValidationLSTMPredictedResidualsVSGroundTruth.png")

  residual_diff_percentages = []
  residual_diff =  torch.abs(decoder_output_seq - test_input_seq)

  # print(len(y_test_dates))
  # print(differences.shape)
  print(torch.max(residual_diff))
  print(torch.min(residual_diff))

  # print(residual_diff_percentages)

  condition = (residual_diff > 0.15)

  filtered_tensor = torch.where(condition, 1, 0)

  print(filtered_tensor.shape)

  filtered_array = filtered_tensor.numpy()


  anomaly_dates = []


  for i in range(len(filtered_array)):
    if filtered_array[i] == 1:
      print(amzn_post_split[i])
      anomaly_dates.append(residuals_test_dates[i])

  #'2023-09-28', '2023-01-05', '2022-07-06', '2021-09-21'

  print(len(anomaly_dates))
  print(anomaly_dates)
  # print(anomaly_dates)

  print(len(residual_diff))
  # print(torch.max(residual_diff_percentages))