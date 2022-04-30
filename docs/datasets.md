# Datasets

`scikit-portfolio` contains a small number of datasets, mostly used for testing and demonstration purpose.
Once loaded they are cached in the home directory under the `.skportfolio` hidden folder, similarly to what is done with `seaborn`.

## Technological stock prices
This small dataset contains almost five years of adjusted closing prices of technological stocks: Apple (AAPL), Microsoft (MSFT), Tesla (TSLA), Amazon (AMZN) and Microstrategy (MSTR).
The data have been obtained through the excellent Python package [yfinance](https://pypi.org/project/yfinance/)

## Simulated normal returns
These simulated returns are obtained from the Matlab financial toolbox, with random seed set to 42.

```textmate
rng(42)
m = [ 0.05; 0.1; 0.12; 0.18 ];
C = [ 0.0064 0.00408 0.00192 0; 
    0.00408 0.0289 0.0204 0.0119;
    0.00192 0.0204 0.0576 0.0336;
    0 0.0119 0.0336 0.1225 ];
m = m/12;
C = C/12;

AssetScenarios = mvnrnd(m, C, 20000);
```

This tool is very useful for testing against commercial implementations of various portfolio optimization methods, see for example the tests in [efficient MAD](efficient_mad.md)

## SP500 prices
SP500 index price

## NASDAQ100 prices
NASDAQ100 prices