import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
import LSTMpy as lstm

## Import nessesary packages
import statsmodels.api as sm
import statsmodels.stats.diagnostic
from tabulate import tabulate
import seaborn as sns
from scipy import stats
import warnings

column = "SEKDKK"

# Data normalization and sliding windows
def sliding_windows(data, seq_length):
    '''
    Function for sliding windows
    '''
    
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


# Preprocessing function for the data + splitting into train and test and scaling


def preprocessing(data, seq_length, train_ratio, val_ratio):
    '''
    Function for preprocessing data
    '''
    
    ## Normalize data
    sc = MinMaxScaler()
    data_normalized = sc.fit_transform(data)
    
    ## Split data into train, validation and test
    x, y = sliding_windows(data_normalized, seq_length)
    
    train_size = int(len(y) * train_ratio)
    val_size = int(len(y) * val_ratio)
    test_size = len(y) - train_size - val_size
    
    x, y = torch.Tensor(x), torch.Tensor(y)
    
    trainX, trainY = x[:train_size], y[:train_size]
    valX, valY = x[train_size : train_size + val_size], y[train_size : train_size + val_size]
    testX, testY = x[train_size + val_size :], y[train_size + val_size :]
    
    return trainX, trainY, valX, valY, testX, testY, sc, train_size, val_size, test_size, y, data_normalized



## Create function for metrics


def metrics(testY_plot, test_predict):
    '''
    Function for metrics
    '''
    
    root_mean_squared_error = np.sqrt(mean_squared_error(testY_plot, test_predict))
    
    print("Test MSE:", mean_squared_error(testY_plot, test_predict))
    print("Test R^2:", r2_score(testY_plot, test_predict))
    print("Test MAE:", mean_absolute_error(testY_plot, test_predict))
    print("Test Median AE:", median_absolute_error(testY_plot, test_predict))
    print("Test MAPE:", mean_absolute_percentage_error(testY_plot, test_predict))
    
    ## output the metrics in a latex table using tabulate
    table = [["R^2", r2_score(testY_plot, test_predict)], 
            ["MSE", mean_squared_error(testY_plot, test_predict)],
            ["MAE", mean_absolute_error(testY_plot, test_predict)],
            ["MAPE", mean_absolute_percentage_error(testY_plot, test_predict)],
            ["RMSE", root_mean_squared_error]]
    
    print(tabulate(table, tablefmt="latex", headers=["Metric", "Value"]))
    
    ## save the table as a latex file
    with open(f"LSTM metrics for SEKDKK.tex", "w") as f:
        f.write(tabulate(table, tablefmt="latex"))
        


def plot_curve200(train_losses, val_losses):
    '''
    Function for plotting the curve for 200 days
    '''

    plt.figure(figsize=(12, 8))
    plt.plot(train_losses[:200], label="Training Loss")
    plt.plot(val_losses[:200], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Losses 200 Epochs – Learning Curve for SEKDKK")
    plt.legend(loc="best", fontsize = 15)
    plt.savefig(f'LSTM learning curve (100 Epochs) for SEKDKK')
    plt.show()
    
    return plt



#Plot the loss and validation loss on the y-axis and the number of epochs on the x-axis
#PLot only for the first 100


def plot_curve_all(train_losses, val_losses):
    '''
    Function for plotting the curve for all epochs
    '''
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Losses – Learning Curve for SEKDKK")
    plt.legend(loc="best", fontsize = 15)
    plt.savefig(f'LSTM learning curve for SEKDKK')
    plt.show()
    
    
## Define the function for "Directional Prediction Accuracy (DPA)"

def DPA(testY_plot, test_predict):
    '''
    Function for calculating the directional prediction accuracy
    '''
    ## Define counter for the number of times the model predicts the correct direction of the exchange rate
    counter = 0
    correct_direction = 0
    false_direction = 0

    ## Loop through the test set and compare the predicted direction with the actual direction
    for i in range(1, (len(test_predict))):
        if test_predict[i] > test_predict[i-1] and testY_plot[i] > testY_plot[i-1]:
            counter += 1
            correct_direction += 1
        elif test_predict[i] < test_predict[i-1] and testY_plot[i] < testY_plot[i-1]:
            counter += 1
            correct_direction += 1
        else:
            counter -= 1
            false_direction += 1
        
    ## output the metrics in a latex table using tabulate and a title 

    table = [["Counter", counter],
                ["Total number of predictions", (len(test_predict)-1)],
                ["Number of correct directions", correct_direction],
                ["Number of false directions", false_direction],
                ["Percentage of correct directions", correct_direction / (len(test_predict)-1)],
                ["Percentage of false directions", false_direction /(len(test_predict)-1)]]


    print(tabulate(table, tablefmt="latex", headers=[f"LSTM {column}"]))
    


def model_eval(trainY_plot, train_predict, testY_plot, test_predict, train_size, val_size, test_size):
    # Plotting
    plt.figure(figsize=(15, 10))
    plt.plot(trainY_plot, label="Training Data")
    plt.plot(range(train_size, train_size + val_size), train_predict[train_size:], label="Validation Data")
    plt.plot(range(train_size + val_size, train_size + val_size + len(testY_plot)), testY_plot, label="Test Data")
    plt.plot(range(train_size + val_size, train_size + val_size + len(test_predict)), test_predict, label="Test Predictions")
    plt.axvline(x=train_size, c="orange", linestyle="--", label="Training-Validation Cut-off")
    plt.axvline(x=train_size + val_size, c="r", linestyle="--", label="Validation-Test Cut-off")
    plt.legend(loc="best", fontsize="x-large")
    plt.title(f"LSTM time-series prediction for SEKDKK", size=20)
    plt.xlabel("Days", size=20)
    plt.ylabel("SEK/DKK", size=20)
    ## save the plot as a png file
    plt.savefig(f'LSTM prediction for SEKDKK')
    plt.show()

    # Plot the last 100 values for the test set
    plt.figure(figsize=(15, 10))
    plt.plot(testY_plot[-100:], label="Test Data")
    plt.plot(test_predict[-100:], label="Test Predictions")
    plt.legend(loc="best", fontsize="x-large")
    plt.title(f"LSTM time-series prediction for SEKDKK (Last 100 Days)" , size=20)
    plt.xlabel("Days", size=20)
    plt.ylabel("SEK/DKK", size=20)
    ## save the plot as a png file
    #plt.savefig(f'LSTM prediction for {column}')
    plt.show()


def jarque(testY_plot, test_predict):
    '''
    Function for calculating the Jarque-Bera test for normality of the residuals
    '''
    residuals =  testY_plot - test_predict

    jb_test = stats.jarque_bera(residuals)
    print("Jarque-Bera test for normality of the residuals: \n \n", jb_test)
    
    return jb_test



def dist(testY_plot, test_predict):
    # Plot distribution of data 
    plt.figure(figsize=(20,10))
    plt.title(f"LSTM – Distribution of model for {column}", size = 20)

    residuals =  testY_plot - test_predict
    mean = residuals.mean()
    mean = float(mean)
    
    # remove warnings
    warnings.filterwarnings('ignore')

    ## add a vertical line at the mean
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label = "Mean")

    ## add true bell curve to the plot
    sns.distplot(residuals, hist=True, kde=True, bins=30, color='b', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, label = "True Distribution")

    plt.legend(fontsize = 15)
    plt.show()
    
    
def summary_table(testY_plot, test_predict, residuals):
    """Function to create a summary table for the model"""
    
    mean = residuals.mean()
    mean = float(mean)
    
    
    
    # define the summary table
    summary_table = [["Mean", mean],
                    ["Standard Error", np.std(residuals)],
                    ["Skewness", stats.skew(residuals)],
                    ["Kurtosis", stats.kurtosis(residuals)],
                    ["Jarque-Bera test", stats.jarque_bera(residuals)[0]],
                    ["p-value", stats.jarque_bera(residuals)[1]],
                    ["Ljung-Box test", statsmodels.stats.diagnostic.acorr_ljungbox(residuals, lags=[5], return_df=True).iloc[0,0]],
                    ["p-value", statsmodels.stats.diagnostic.acorr_ljungbox(residuals, lags=[5], return_df=True).iloc[0,1]],
                    ["Heteroscedasticity", statsmodels.stats.diagnostic.het_arch(residuals)[0]],
                    ["p-value", statsmodels.stats.diagnostic.het_arch(residuals)[1]],
                    ["Normality", stats.normaltest(residuals)[0]],
                    ["p-value", stats.normaltest(residuals)[1]]]
    
    
    print(tabulate(summary_table, tablefmt="latex", headers=[f"Summary LSTM – {column}", "Value"]))