# EE596 Final Project
## Contributors
Ayush Agarwal and Pranav Chunduru

## Project Overview
The stock market is highly influenced by global events, economic shifts, and company-specific developments. These factors can lead to significant fluctuations in stock prices, reflecting critical financial events. This project aims to identify such anomalies in stock prices by analyzing the historical data of a specific stock.

The primary objective is to predict anomalies—large deviations in stock prices—that may signal major events, either positive or negative, for a particular stock. By identifying these anomalies, we hope to enhance the understanding of market trends and provide insights that could help mitigate risks or capitalize on emerging opportunities. While our current scope is focused on individual stocks, an extension to broader market analysis remains a potential stretch goal.

## Prerequisites and Setup
### Required Libraries
This project relies on the following Python libraries, which are also listed in the `requirements.txt` file:

* `pandas`: For reading and manipulating stock price data from CSV files.
* `numpy`: For efficient array operations.
* `torch`: To implement and train Long Short-Term Memory (LSTM) models.
* `scikit-learn`: For scaling and preprocessing data.
* `matplotlib`: For data visualization and plotting.

### Installation Instructions
To install the required dependencies, ensure you have Python and `pip` installed on your system. If you are using macOS, you can execute the following command:

```pip install -U pandas numpy torch torchvision scikit-learn matplotlib```

This command installs the necessary libraries and prepares your environment for running the project.

## Running the Project
### Workflow Overview
The project consists of three main Python scripts:

* `model.py`: Contains the implementation of the machine learning models.
* `utils.py`: Provides utility functions for preprocessing and post-processing data.
* `main.py`: Integrates the models and utilities, performing training and evaluation.

### Step-by-Step Instructions
Run `main.py` to train the models. On the first execution, this script will create and save two models in the checkpoints/ directory:

* `Encoder_Decoder_RNN`: Predicts stock prices for Amazon (AMZN).
* `Encoder_Decoder_RNN_Residual`: Predicts residuals to refine the main model's predictions.

Once the models are trained, you can execute `demo.py` to demonstrate the predictions. This script uses the saved models to analyze stock price data for AMZN, starting from their 20:1 stock split on June 6, 2022.

### Backup Option
In case the above step-by-step instructions fail to successfully run the project. There is a backup option. You may run `demo_full.py` which compiles all the code in the workflow into one large `.py` and will execute if run with the above dependencies installed.

**NOTE:** This should be treated as a backup option if there is something wrong with the other files.

## Expected Output
The demo.py script generates variable outputs during execution. Ultimately, it produces a list of dates representing predicted anomalies in stock prices. These dates indicate significant deviations between actual stock prices and model predictions. A sample output might look like this:

```['2023-09-28', '2023-01-05', '2022-07-06', '2021-09-21']```

These dates are associated with potentially critical financial events for the stock being analyzed.

# Dataset and Acknowledgments
This project utilizes a publicly available dataset hosted on Hugging Face. The dataset provides comprehensive historical stock price data essential for training and evaluating our models.

* Dataset URL: Hugging Face - FNSPID Dataset

We extend our gratitude to the dataset creator for making this valuable resource publicly accessible.

