# BSc in Economics – University of Copenhagen

## Comparative Analysis of the ARIMA and LSTM Models, for Exchange Rate Prediction
### A Case Study on the Swedish-Danish Currency Pair (Econometrics)

![LSTM prediction](/Users/vg/Desktop/BSc.-Economics-Comparative-Analysis-of-the-ARIMA-and-LSTM-Models-for-Exchange-Rate-Prediction./Model – BA/LSTM SEKDKK/LSTM prediction for SEKDKK.png)

**Members**
| Name | Student ID |
| --- | --- |
| Markus Kyhl Jacobi | rdq254 |
| Valdemar Rusbjerg Guldager | gvz104 |


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
Data is taken from Sveriges Riksbank. The data is downloaded from the following link: https://www.riksbank.se/en-gb/statistics/search-interest--exchange-rates/interest-rates-and-currencies/official-exchange-rates/ and is stored in the `20_year_data.csv` file. The data is stored in the following format: Comma Seperated Values.

## Main
This section describes the main files for the project.
The main files are structured in the folder MODEL - BA as follows:

1. LSTM MODEL FINAL.ipynb
2. ARIMA MODEL FINAL.ipynb
3. LSTMpy.py
4. ARIMA.py
5. 20_year_data.csv

#### Language
- Language used: Python 3.9.16


### LSTM
THe main file for the LSTM model is `LSTM MODEL FINAL.ipynb`. 
The file is dependent on the file `LSTMpy.py` which contains functions for the LSTM model.
- Short function description is found in the file `LSTMpy.py`.


### ARIMA
The main file for the ARIMA model is `ARIMA MODEL FINAL.ipynb`.
The file is dependent on the file `ARIMA.py` which contains functions for the ARIMA model.
- Short function description is found in the file `ARIMA.py`.



## NOTE!!
- The "SAFEGUARD" files are only for backup and 
- Possible that pictures are not true to the pictures used in the original thesis – as some pictured haven't been changed.





