# Comparative forecasting of financial time series_replication

A reproducible R project comparing classical time-series models and tree-based machine-learning models for daily NIFTY 50 closing-price forecasting.

## Repository contents

- `nifty50_complete_analysis.R` — end-to-end analysis: Yahoo Finance data download, log-price/log-return construction, exploratory plots, stationarity and dependence tests, ARMA order selection using AIC/BIC, residual diagnostics, ARMA–GARCH estimation and diagnostics, rolling forecasts, Random Forest and XGBoost tuning, test predictions, metrics, and plots.
- `Results/` — recorded result tables from the experiment discussed in the report.

The R script generates the PNG figures and CSV outputs in the working directory when it is run. The figure filenames correspond to those referenced by the LaTeX report. Run the R script first, then compile the `.tex` file from the project root so the figure files are available. The compiled PDF and generated PNGs are not committed in this initial repository version; they can be recreated from the included source files.

## Methods included

1. Exploratory analysis of closing prices, log prices, daily log returns, first-differenced log returns, rolling statistics, ACF and PACF.
2. ADF, KPSS and Phillips–Perron stationarity tests; Ljung–Box and ARCH-LM dependence diagnostics.
3. ARMA order search over `p, q = 0,...,10`, with and without a mean term, selected using AIC and BIC. The selected return models correspond to ARIMA(5,1,5) with drift and ARIMA(0,1,0) without drift on log prices.
4. Four ARMA–GARCH(1,1) specifications: ARMA(5,5) and ARMA(0,0), each with Normal and Student-t innovations.
5. Random Forest regression with five lagged returns, tuning `mtry` and `nodesize` while keeping `ntree = 500`.
6. XGBoost regression with five lagged returns, tuning `max_depth`, `eta`, `min_child_weight`, and boosting rounds.
7. Rolling one-step-ahead price forecasts and a common 500-observation test-set comparison using ME, MAE, RMSE and MAPE.

## Data

The script downloads the Yahoo Finance ticker `^NSEI` using `quantmod`. The requested start date is 1995-11-03; in the experiment documented here, Yahoo Finance returned observations beginning on 2007-09-18. The script saves the downloaded closing-price snapshot as `NIFTY50_Closing_Prices.csv` each time it runs.

Yahoo Finance data can be updated or revised. For an exact reproduction, use the same download end date as the original experiment. The script accepts an optional environment variable:

```r
Sys.setenv(NIFTY_DOWNLOAD_TO = "YYYY-MM-DD")
source("nifty50_complete_analysis.R")
```

Replace `YYYY-MM-DD` with the original download end date. If the variable is not set, the script uses `Sys.Date()`.

## How to run

1. Install R (4.x recommended) and RStudio.
2. Clone or download this repository.
3. Open the project folder as the R working directory.
4. Run:

```r
source("nifty50_complete_analysis.R")
```

The script installs missing packages and loads the following packages: `quantmod`, `forecast`, `tseries`, `FinTS`, `rugarch`, `randomForest`, `xgboost`, `ggplot2`, and `zoo`.

The full ARMA grid search, GARCH estimation, and XGBoost grid search can take some time. Internet access is required to retrieve Yahoo Finance data.

## Main reported results

Price-level metrics on the 500-observation test set (lower MAE, RMSE and MAPE are better):

| Model | ME | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|---:|
| ARIMA(5,1,5) | -8.2379 | 154.5298 | 205.2892 | 0.6412 |
| ARIMA(0,1,0) | 1.2911 | **151.6817** | **202.3063** | **0.6290** |
| ARMA(5,5)–GARCH Normal | -16.1375 | 154.1686 | 203.7747 | 0.6397 |
| ARMA(5,5)–GARCH Student-t | -17.3660 | 153.9794 | 203.7966 | 0.6388 |
| ARMA(0,0)–GARCH Normal | 1.2911 | 151.6817 | 202.3063 | 0.6290 |
| ARMA(0,0)–GARCH Student-t | 1.2911 | 151.6817 | 202.3063 | 0.6290 |
| Tuned Random Forest | -5.4442 | 156.5184 | 205.2712 | 0.6488 |
| Tuned XGBoost | -7.9509 | 152.2220 | 202.5839 | 0.6314 |

Selected machine-learning configurations:

- Random Forest: `ntree = 500`, `mtry = 1`, `nodesize = 10`.
- XGBoost: `max_depth = 2`, `eta = 0.03`, `min_child_weight = 5`, `nrounds = 5`. Five rounds were chosen as a near-equivalent validation alternative to the one-round minimum.

## Important methodological notes

- The forecasting target is the ordinary daily log return, `diff(log(close))`, not the first difference of returns.
- Each price forecast is reconstructed as `previous_actual_close * exp(predicted_return)`. This is a rolling one-step-ahead evaluation, not a recursively accumulated 500-step path.
- Model parameters are estimated on the training data and kept fixed during the test period; newly observed returns are used as they become available.
- The machine-learning validation split is chronological and is taken from the training sample. The final test set is reserved for evaluation.
- The ARMA(0,0) zero-mean and the corresponding ARMA(0,0)–GARCH specifications have identical point-price forecasts because their predicted conditional mean is zero. GARCH changes the conditional volatility forecast, not the mean forecast.
- In the recorded diagnostics, the GARCH models passed the reported squared-residual and ARCH-LM checks, but stability, sign-bias and Pearson goodness-of-fit tests raised concerns. Stationarity-test statistics should be verified against the exact input series before using them in a formal paper.

## Outputs generated by the R script

Examples include:

- `NIFTY50_Closing_Prices.csv`
- `ARMA_Grid_Search_Results.csv`
- `GARCH_Information_Criteria.csv`
- `Random_Forest_Grid_Search.csv`
- `XGBoost_Grid_Search.csv`
- `NIFTY50_Forecast_Metrics.csv`
- `NIFTY50_Test_Predictions.csv`
- Actual-versus-predicted plots for ARIMA, GARCH, Random Forest and XGBoost, plus a combined plot.

## Reproducibility and interpretation

The reported results are specific to the downloaded data vintage and the selected 500-observation test window. Re-run the script using the original data cutoff to reproduce the numerical results. Results may differ when the download date, historical Yahoo Finance data, package versions, or tuning environment changes.

This repository is for research and educational purposes only; it is not investment advice.
