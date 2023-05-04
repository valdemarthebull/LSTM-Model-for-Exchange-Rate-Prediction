## This script is used to generate the ARIMA model for the time series data
# The script defines the functions used for the ARIMA jupyter file.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tabulate import tabulate
import LSTMpy as lstm
import warnings
import tqdm

## White Noise Test for the data
from statsmodels.stats.diagnostic import acorr_ljungbox

## set random seed
np.random.seed(42)


def test_stationarity(timeseries, column, when=''):
    '''
    This function is used to test the stationarity of the time series data.
    The function uses the Augmented Dickey-Fuller test to test the stationarity of the data.
    The function outputs a table of the results and saves the table as a latex table.
    '''
    
    
    result = adfuller(timeseries)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value}")
    if result[1] > 0.05:
        print("The series is not stationary")
   
    else:
        print("The series is stationary")
    
    ## use tabulate to output the results in a latex table
    table = [["ADF Statistic", result[0]], ["p-value", result[1]]]
    for key, value in result[4].items():
        table.append([key, value])
    
    ## Add line to say if the series is stationary or not
    if result[1] > 0.05:
        table.append(["The series is not stationary as the p-value is greater than 0.05"])
    else:
        table.append(["The series is stationary as the p-value is less than 0.05"])
    
    print(tabulate(table, tablefmt="latex"))
    
    ## save the table a latex table to the path chosen above
    with open(f"ARIMA {column}_stationarity({when}).tex", "w") as f:
        f.write(tabulate(table, tablefmt="latex"))



## differencing function and removing outliers

def diff_outliers(data, column):
    '''
    This function is used to difference the data and remove the outliers.
    The function outputs a table of the results and saves the table as a latex table.
    '''
    
    # 1st order differencing to make the series stationary
    diff_data = data.diff().dropna()
    
    # Remove outliers
    diff_data = diff_data[diff_data < 0.05]
    diff_data = diff_data[diff_data > -0.05]
    
    # table of the removed outliers
    table = [["Removed outliers", len(data) - len(diff_data)], ["Total number of observations", len(data)]]
    print(tabulate(table, tablefmt="latex"))
    
    ## save the table a latex table
    with open(f"ARIMA {column}_outliers.tex", "w") as f:
        f.write(tabulate(table, tablefmt="latex"))
    
    return diff_data



def distribution(diff_data, column):
    '''
    This function is used to plot the distribution of the data.
    The function outputs a plot of the distribution and saves the plot as a png.
    '''
    
    # Calculate the mean of the data
    mean = diff_data.mean()
    mean = float(mean)
    
    # Plot distribution of data 
    plt.figure(figsize=(20,10))
    plt.title(f"Distribution of data for {column}", size = 20)
    
    ## add a vertical line at the mean
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label = "Mean")
    
    ## add legend to plot
    
    
    ## add true bell curve to the plot
    sns.distplot(diff_data, hist=True, kde=True, bins=30, color='b', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}, label = "True Distribution")
    
    plt.legend(fontsize = 15)
    
    ## Save fig to path
    plt.savefig(f"ARIMA {column}_distribution.png")
    
    plt.show()
    return mean


def plot_all(data_train, data_test, diff_data, column, mean):
    '''
    This function is used to plot the train and test data.
    The function outputs a plot of the train and test data and saves the plot as a png.
    '''
    
    plt.figure(figsize=(20, 10))
    plt.plot(data_train[-1000:], label = 'Train Data (Last 1000 days)')
    plt.plot(data_test, label = 'Test Data')
    plt.title('Train (Only 200 days) and Test data')
    plt.axvline(x=data_train.index[-1],color='black', linestyle='--', label = 'Train/Test data cut-off')
    plt.xticks(rotation=45)
    plt.title(f'Train and Test data for {column}', size = 30)
    mean = diff_data.mean()
    mean = float(mean)
    print(mean)
    plt.axhline(mean, color='r', linestyle='--', label = 'Data Mean')
    plt.legend(loc='best', fontsize = 15)
    plt.savefig(f"ARIMA {column}_train_test.png")
    plt.show()


def white_noise(data, lags, name=''):
    """ 
    Function to test if the data is white noise
    """
    ## run the test
    test_results = sm.stats.acorr_ljungbox(data, lags=[lags], return_df=True)
    
    ## extract the p-value from the test results
    p_value = test_results.iloc[0,1]
    
    ## print the results
    print(f"The p-value from the white noise test is {p_value}")
    
    if p_value < 0.05:
        print("The data is not white noise")
    else:
        print("The data is white noise")
        
    ## Output the test in a latex table using tabulate
    table = [["Lag", "p-value"], [lags, p_value]]
    print(tabulate(table, tablefmt="latex", headers = name))

    
    
    ## return the p-value
    return p_value


## Initialising the model

def model_initialise(data_train, data_test, column, mean, diff_data):
    '''
    This function is used to initialise the model.
    The function outputs a plot of the train and test data and saves the plot as a png.
    '''
    
    ## Create a matrix of zeros (Possible to use WLS aswell)
    aic_matrix = np.zeros((10,10))
    warnings.filterwarnings("ignore")
    
    
    ## Loop through all the p and q values and fit the model and use progress bar to show progress from tqdm
    for p in tqdm.tqdm(range(10)):
        for q in range(10):
            try:
                model = sm.tsa.ARIMA(diff_data, order=(p,1,q)).fit()
                aic_matrix[p,q] = model.aic
            except:
                pass
    
    ## Find the p and q values that give the lowest AIC value
    p,q = np.unravel_index(np.argmin(aic_matrix), aic_matrix.shape)
    
    ## Print the p and q values
    print(f"The p and q values are {p} and {q} respectively")
    
    ## Print the AIC value
    print(f"The AIC value is {aic_matrix[p,q]}")
    
    ## Fit the model with the p and q values
    model = sm.tsa.ARIMA(diff_data, order=(p,1,q)).fit()
    
    ## Print the model summary
    print(model.summary())
    
    ## Save the model summary to a latex table
    with open(f"ARIMA {column}_model_summary.tex", "w") as f:
        f.write(model.summary().as_latex())
    
    return model, aic_matrix, p, q
    

## Function for heatmap of AIC values

def heatmap(aic_matrix, column):
    '''
    This function is used to plot the heatmap of the AIC values.
    The function outputs a plot of the heatmap and saves the plot as a png.
    '''
    # change size to 20 X 10 inches
    plt.figure(figsize=(20,10))

    ## make heat map only show the 10 best AIC values
    sns.heatmap(aic_matrix, mask=(aic_matrix==0), annot=True, fmt='.2f', cmap = 'coolwarm')
    plt.title(f'AIC Heatmap for different p and q values [Lower is better] – {column}', size = 20 )
    plt.xlabel('q', size = 20)
    plt.ylabel('p', size = 20)

    ## Save fig to path
    plt.savefig(f"ARIMA {column}_aic_heatmap.png")

    plt.show()

    print("The best p and q values are")
    print(np.where(aic_matrix == np.min(aic_matrix)))


## Plot the 30-day forecast and compare it to the actual values
def plot_forecast_comparison(model, y_train, y_test, column, mean, steps=210):
    forecast = model.predict(n_periods=steps)
    forecast = pd.Series(forecast, index=y_test.index)
    
    plt.figure(figsize=(20, 10))
    plt.plot(y_train[-400:], label='Observed (Training) (400 days)')
    plt.plot(y_test, label='Observed (Test)')
    plt.plot(forecast, label='Forecast')
    plt.axvline(x=y_train.index[-1],color='black', linestyle='--', label = 'Train/Test data cut-off')
    plt.title(f'Forecast vs Actual for {column}', size = 30)
    plt.axhline(mean, color='r', linestyle='--', label = 'Data Mean')
    plt.xlabel('Date')
    plt.ylabel('SEKDKK', size = 20)
    plt.legend(fontsize = 15)
    
    ## save fig to path
    plt.savefig(f"ARIMA {column}_forecast.png")
    
    plt.show()
    
    return forecast
    
    
    
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    ## output in a latex table
    table = [["R2", r2], ["MSE", mse], ["MAE", mae], ["MAPE", mape], ["RMSE", rmse]]
    print(tabulate(table, tablefmt="latex"))
    
    return table



def direction_counter(forecast, data_test, column):

    ## Define counter for the number of times the model predicts the correct direction of the exchange rate
    counter = 0
    counter_random = 0

    # initiate random walk for comparison
    correct_direction_random = 0
    false_direction_random = 0

    # initiate model direction counter
    correct_direction = 0
    false_direction = 0

    ## initiate random walk for comparison
    # set seed
    np.random.seed(42)

    # create random walk with forecast as index
    random_walk = np.cumsum(np.random.normal(0, 1, len(forecast)))

    # set index for random walk to be the same as the forecast
    random_walk = pd.Series(random_walk, index=forecast.index)


    # create a data frame with forecast, random walk and actual values
    ## create a data frame with forecast, random walk and actual values
    df = pd.DataFrame({'forecast': forecast, 'random_walk': random_walk, 'actual': data_test.squeeze()})


    ## create loop to check if the forecast predicts the correct direction of the exchange rate when compared to the actual values
    for i in range((len(df)-1)):
        if df['forecast'][i] > df['forecast'][i-1] and df['actual'][i] > df['actual'][i-1]:
            counter += 1
            correct_direction += 1
        elif df['forecast'][i] < df['forecast'][i-1] and df['actual'][i] < df['actual'][i-1]:
            counter += 1
            correct_direction += 1
        else:
            counter += 1
            false_direction += 1
            
        if df['random_walk'][i] > df['random_walk'][i-1] and df['actual'][i] > df['actual'][i-1]:
            counter_random += 1
            correct_direction_random += 1
        elif df['random_walk'][i] < df['random_walk'][i-1] and df['actual'][i] < df['actual'][i-1]:
            counter_random += 1
            correct_direction_random += 1
        else:
            counter_random += 1
            false_direction_random += 1


    ## create latex table for the number of times the model predicts the correct direction of the exchange rate
    table_model = [["Counter", counter],
                ["Total number of predictions", len(forecast)],
                ["Number of correct directions", correct_direction],
                ["Number of false directions", false_direction],
                ["Percentage of correct directions", correct_direction / len(forecast)],
                ["Percentage of false directions", false_direction / len(forecast)]]

    #print(tabulate(table_model, tablefmt="latex"))

    table_random = [["Counter", counter_random],
                ["Total number of predictions", len(forecast)],
                ["Number of correct directions", correct_direction_random],
                ["Number of false directions", false_direction_random],
                ["Percentage of correct directions", correct_direction_random / len(forecast)],
                ["Percentage of false directions", false_direction_random / len(forecast)]]

    #print(tabulate(table_random, tablefmt="latex"))

    ## a table which includes direction metrics for the random walk and the model with titles for random walk and model

    pct_correct_arima = (correct_direction / (len(forecast) - 1))
    pct_correct_random = (correct_direction_random / (len(forecast) - 1))
    pct_wrong_arima = 1 - pct_correct_arima
    pct_wrong_random = 1 - pct_correct_random


    table_concat = [
                ["Total number of predictions", len(forecast) - 1, len(forecast) - 1],
                ["Number of correct directions", correct_direction, correct_direction_random],
                ["Number of false directions", false_direction, false_direction_random],
                ["Percentage of correct directions", "{:.2%}".format(pct_correct_arima), "{:.2%}".format(pct_correct_random)],
                ["Percentage of false directions", "{:.2%}".format(pct_wrong_arima), "{:.2%}".format(pct_wrong_random)]]

    print(tabulate(table_concat, tablefmt="latex", headers=[f"ARIMA {column}", "Random walk"]))
            

    ## save table to latex with, title headers and name
    name = [f"ARIMA {column}", "Random walk"]

    with open(f"ARIMA {column}_direction.tex", "w") as f:
        f.write(tabulate(table_concat, tablefmt="latex", headers = name))

    return 


def rand(random_walk, forecast, data_test, column):
    '''
    Function to create a random walk and compare it to the actual values and the forecast
    '''

    
    df = pd.DataFrame({'forecast': forecast, 'random_walk': random_walk, 'actual': data_test.squeeze()})
    df = df.dropna()

    # create table for last 30 days for random walk and print model in latex
    table = df[-30:].describe()

    # see if random walk can predict the direction of the exchange rate of the last 30 days for each day
    counter = 0
    correct_direction = 0
    false_direction = 0

    for i in range(len(df[-30:])):
        if df['random_walk'][i] > df['random_walk'][i-1] and df['actual'][i] > df['actual'][i-1]:
            counter += 1
            correct_direction += 1
        elif df['random_walk'][i] < df['random_walk'][i-1] and df['actual'][i] < df['actual'][i-1]:
            counter += 1
            correct_direction += 1
        else:
            counter += 1
            false_direction += 1
            
    pct_correct_random = (correct_direction / (len(df[-30:])))
    pct_wrong_random = 1 - pct_correct_random

    table_random = [["Counter", counter],
                ["Total number of predictions", len(df[-30:])],
                ["Number of correct directions", correct_direction],
                ["Number of false directions", false_direction],
                ["Percentage of correct directions", "{:.2%}".format(pct_correct_random)],
                ["Percentage of false directions", "{:.2%}".format(pct_wrong_random)]]


    # print table with headers
    print(tabulate(table_random, tablefmt="latex", headers=["Random walk"]))
    


def actual_forecat_random(random_walk, data_test, column, mean, model, y_test, steps = 30):
    '''
    Function to plot the actual values, the forecast and the random walk
    '''
    
    
    # Plot the last 30 values for the test set with random walk on second y axis
    plt.figure(figsize=(20, 10))

    # define two y axes
    fig, ax1 = plt.subplots()

    # make subplot bigger
    fig.set_size_inches(20, 10)
    
    forecast = model.predict(n_periods=steps)
    forecast = pd.Series(forecast, index=y_test.index)
    

    # plot test data and predictions on first y axis
    ax1.plot(data_test[-30:], label='Actual', color = 'blue')
    ax1.plot(forecast, label='Forecast', color = 'orange')
    ax1.set_title(f'ARIMA – Forecast vs Actual for {column}', size = 30)
    ax1.axhline(mean, color='r', linestyle='--', label = 'Data Mean')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('SEKDKK', size = 20)

    # plot random walk on second y axis
    ax2 = ax1.twinx()
    ax2.plot(random_walk, color='green', label='Random Walk')
    ax2.set_ylabel('Random walk', size = 20)
    ax2.tick_params(axis='y')


    ## make legend for both y axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc=0, fontsize = 20)

    # save plot
    plt.savefig(f"ARIMA {column}_forecast(30)_random.png")

    plt.show()