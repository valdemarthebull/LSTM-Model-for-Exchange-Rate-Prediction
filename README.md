# BSc in Economics – University of Copenhagen

## Comparative Analysis of the ARIMA and LSTM Models, for Exchange Rate Prediction.
### A Case Study on the Swedish-Danish Currency Pair. (Econometrics)
\

**Members**
- Markus Kyhl Jacobi            rdq254
- Valdemar Rusbjerg Guldager    gvz104


## Libraries
The following libraries are used in the project:
- **numpy** for numerical calculations.
- **pandas** for data manipulation.
- **matplotlib** for plotting.
- **seaborn** for plotting.
- **sklearn** for preprocessing and metrics.
- **statsmodels** for ARIMA.
- **pytorch** for LSTM and neural network build.
- **tqdm** for progress bar.
- **datetime** for date manipulation.
- **warnings** for ignoring warnings.

### Installation
The libraries can be installed using the following command in the terminal for the requirements.txt file:
```
pip install -r requirements.txt
```

## Data
Data is taken from Sveriges Riksbank. The data is downloaded from the following link: https://www.riksbank.se/en-gb/statistics/search-interest--exchange-rates/interest-rates-and-currencies/official-exchange-rates/ and is stored in the `29_year_data.csv` file. The data is stored in the following format: Comma Seperated Values.

## Main
This section describes the main files for the project. The main files are structured as follows:
1. LSTM
2. ARIMA

### LSTM
THe main file for the LSTM model is `LSTM MODEL.ipynb`. 
The file is dependent on the file `LSTMpy.py` which contains functions for the LSTM model.


### ARIMA
The main file for the ARIMA model is `ARIMA MODEL TRUE.ipynb`.
The file is dependent on the file `ARIMA.py` which contains functions for the ARIMA model.



## NOTE!
The "Safeguard" file is only for backup and pictures are not the correct pictures used in the original thesis.





