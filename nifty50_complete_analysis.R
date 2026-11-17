# ============================================================
# NIFTY 50: Complete Forecasting Analysis
# ARIMA, ARMA-GARCH, Random Forest and XGBoost
# ============================================================

rm(list = ls())
options(stringsAsFactors = FALSE, scipen = 999)
set.seed(123)

# ------------------------------------------------------------
# 1. Install and load required packages
# ------------------------------------------------------------
packages <- c(
  "quantmod", "forecast", "tseries", "FinTS", "rugarch",
  "randomForest", "xgboost", "ggplot2", "zoo"
)

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
}

# ------------------------------------------------------------
# 2. Download NIFTY 50 closing prices
# ------------------------------------------------------------
# For exact reproducibility, replace Sys.Date() with the same
# download end date used to produce the reported results.
download_from <- as.Date("1995-11-03")
download_to_env <- Sys.getenv("NIFTY_DOWNLOAD_TO", unset = "")
download_to <- if (nzchar(download_to_env)) as.Date(download_to_env) else Sys.Date()
if (is.na(download_to)) {
  stop("NIFTY_DOWNLOAD_TO must be a valid date in YYYY-MM-DD format.")
}

data_xts <- quantmod::getSymbols(
  "^NSEI",
  src = "yahoo",
  from = download_from,
  to = download_to,
  auto.assign = FALSE
)

nifty_closing <- na.omit(quantmod::Cl(data_xts))

cat("First available observation:", as.character(index(nifty_closing)[1]), "\n")
cat("Last available observation :", as.character(tail(index(nifty_closing), 1)), "\n")
cat("Number of closing prices    :", NROW(nifty_closing), "\n")

if (NROW(nifty_closing) <= 600) {
  stop("Not enough observations for the selected training/test design.")
}

# ------------------------------------------------------------
# 3. Construct log prices, returns and the train/test split
# ------------------------------------------------------------
log_prices <- log(nifty_closing)
all_returns <- as.numeric(diff(log_prices))
return_dates <- as.Date(index(nifty_closing))[-1]

if (any(!is.finite(all_returns))) {
  stop("Non-finite log returns detected. Check the downloaded prices.")
}

test_size <- 500
n_prices <- NROW(nifty_closing)
train_price_count <- n_prices - test_size

train_close <- nifty_closing[1:train_price_count]
test_close <- nifty_closing[(train_price_count + 1):n_prices]

train_returns <- as.numeric(diff(log(train_close)))
test_returns <- as.numeric(diff(log(c(tail(train_close, 1), test_close))))

actual_test_prices <- as.numeric(test_close)
test_dates <- as.Date(index(test_close))
previous_prices <- c(
  as.numeric(tail(train_close, 1)),
  head(actual_test_prices, -1)
)

stopifnot(length(test_returns) == test_size)
stopifnot(length(actual_test_prices) == test_size)
stopifnot(length(previous_prices) == test_size)

cat("Training return observations:", length(train_returns), "\n")
cat("Testing return observations :", length(test_returns), "\n")

# ------------------------------------------------------------
# 4. Exploratory plots: prices, returns and differenced returns
# ------------------------------------------------------------
price_data <- data.frame(
  Date = as.Date(index(nifty_closing)),
  Closing_Price = as.numeric(nifty_closing)
)

log_price_data <- data.frame(
  Date = as.Date(index(log_prices)),
  Log_Price = as.numeric(log_prices)
)

return_data <- data.frame(
  Date = return_dates,
  Log_Return = all_returns
)

first_diff_returns <- diff(train_returns)
first_diff_dates <- tail(return_dates[seq_along(train_returns)], length(first_diff_returns))
first_diff_data <- data.frame(
  Date = first_diff_dates,
  First_Differenced_Return = first_diff_returns
)

p_price <- ggplot(price_data, aes(Date, Closing_Price)) +
  geom_line(colour = "black", linewidth = 0.5) +
  labs(title = "NIFTY 50 Daily Closing Prices", x = "Date", y = "Closing Price") +
  theme_minimal(base_size = 12)

ggplot2::ggsave("NIFTY50_Closing_Prices.png", p_price, width = 12, height = 5, dpi = 300)

p_logprice <- ggplot(log_price_data, aes(Date, Log_Price)) +
  geom_line(colour = "black", linewidth = 0.5) +
  labs(title = "Logarithm of NIFTY 50 Closing Prices", x = "Date", y = "Log Price") +
  theme_minimal(base_size = 12)

ggplot2::ggsave("NIFTY50_Log_Prices.png", p_logprice, width = 12, height = 5, dpi = 300)

p_returns <- ggplot(return_data, aes(Date, Log_Return)) +
  geom_line(colour = "black", linewidth = 0.4) +
  labs(title = "Daily Log Returns of NIFTY 50", x = "Date", y = "Log Return") +
  theme_minimal(base_size = 12)

ggplot2::ggsave("return.png", p_returns, width = 12, height = 5, dpi = 300)
ggplot2::ggsave("Returns.png", p_returns, width = 12, height = 5, dpi = 300)

p_diff_returns <- ggplot(first_diff_data, aes(Date, First_Differenced_Return)) +
  geom_line(colour = "black", linewidth = 0.4) +
  labs(title = "First Differences of Daily Log Returns",
       x = "Date", y = "First-Differenced Log Return") +
  theme_minimal(base_size = 12)

ggplot2::ggsave("NIFTY50_First_Differenced_Log_Returns.png", p_diff_returns,
                width = 12, height = 5, dpi = 300)

# Rolling statistics are calculated on the training returns.
rolling_window <- 252
rolling_mean <- zoo::rollapply(
  train_returns, width = rolling_window, FUN = mean,
  fill = NA, align = "right"
)
rolling_sd <- zoo::rollapply(
  train_returns, width = rolling_window, FUN = sd,
  fill = NA, align = "right"
)
train_return_dates <- head(return_dates, length(train_returns))
rolling_data <- data.frame(
  Date = train_return_dates,
  Rolling_Mean = as.numeric(rolling_mean),
  Rolling_SD = as.numeric(rolling_sd)
)

p_rolling_mean <- ggplot(rolling_data, aes(Date, Rolling_Mean)) +
  geom_line(colour = "blue", linewidth = 0.5) +
  labs(title = "252-Day Rolling Mean of Log Returns", x = "Date", y = "Rolling Mean") +
  theme_minimal(base_size = 12)

ggplot2::ggsave("NIFTY50_Rolling_Mean.png", p_rolling_mean, width = 12, height = 5, dpi = 300)

p_rolling_sd <- ggplot(rolling_data, aes(Date, Rolling_SD)) +
  geom_line(colour = "blue", linewidth = 0.5) +
  labs(title = "252-Day Rolling Standard Deviation of Log Returns",
       x = "Date", y = "Rolling Standard Deviation") +
  theme_minimal(base_size = 12)

ggplot2::ggsave("NIFTY50_Rolling_SD.png", p_rolling_sd, width = 12, height = 5, dpi = 300)

# ACF and PACF plots of training returns and squared returns.
png("NIFTY50_ACF_Returns.png", width = 1200, height = 800, res = 130)
acf(train_returns, lag.max = 40, main = "ACF of Daily Log Returns")
dev.off()

png("NIFTY50_PACF_Returns.png", width = 1200, height = 800, res = 130)
pacf(train_returns, lag.max = 40, main = "PACF of Daily Log Returns")
dev.off()

png("NIFTY50_ACF_Squared_Returns.png", width = 1200, height = 800, res = 130)
acf(train_returns^2, lag.max = 40, main = "ACF of Squared Daily Log Returns")
dev.off()

# ------------------------------------------------------------
# 5. Stationarity tests: log prices, returns, differenced returns
# ------------------------------------------------------------
run_stationarity_tests <- function(x, series_name) {
  x <- as.numeric(na.omit(x))
  cat("\n============================================\n")
  cat("Stationarity tests for:", series_name, "\n")
  cat("============================================\n")
  cat("ADF test:\n")
  print(tryCatch(tseries::adf.test(x), error = function(e) e))
  cat("KPSS test (level stationarity null):\n")
  print(tryCatch(tseries::kpss.test(x, null = "Level"), error = function(e) e))
  cat("Phillips-Perron test:\n")
  print(tryCatch(tseries::pp.test(x), error = function(e) e))
}

train_log_prices <- as.numeric(log(train_close))
run_stationarity_tests(train_log_prices, "Training log prices")
run_stationarity_tests(train_returns, "Ordinary training log returns")
run_stationarity_tests(first_diff_returns, "First differences of training log returns")

# ------------------------------------------------------------
# 6. Serial-dependence and ARCH tests on raw training returns
# ------------------------------------------------------------
cat("\nLjung-Box tests on raw training returns:\n")
for (lag_value in c(10, 20, 40)) {
  cat("Lag:", lag_value, "\n")
  print(Box.test(train_returns, lag = lag_value, type = "Ljung-Box"))
}

cat("\nLjung-Box test on squared training returns:\n")
print(Box.test(train_returns^2, lag = 20, type = "Ljung-Box"))

cat("\nARCH-LM test on raw training returns:\n")
print(FinTS::ArchTest(train_returns, lags = 12, demean = TRUE))

# ------------------------------------------------------------
# 7. ARMA grid search using AIC and BIC
# ------------------------------------------------------------
# Search p,q = 0,...,10 with and without a mean term.
arma_grid <- expand.grid(
  p = 0:10,
  q = 0:10,
  include_mean = c(TRUE, FALSE)
)

arma_grid$AIC <- NA_real_
arma_grid$BIC <- NA_real_
arma_grid$AICc <- NA_real_
arma_fits <- vector("list", nrow(arma_grid))

for (i in seq_len(nrow(arma_grid))) {
  fit_i <- tryCatch(
    forecast::Arima(
      train_returns,
      order = c(arma_grid$p[i], 0, arma_grid$q[i]),
      include.mean = arma_grid$include_mean[i],
      method = "ML"
    ),
    error = function(e) NULL
  )

  arma_fits[i] <- list(fit_i)

  if (!is.null(fit_i)) {
    arma_grid$AIC[i] <- fit_i$aic
    arma_grid$BIC[i] <- fit_i$bic
    arma_grid$AICc[i] <- fit_i$aicc
  }
}

arma_grid_valid <- arma_grid[is.finite(arma_grid$AIC) & is.finite(arma_grid$BIC), ]
arma_grid_aic <- arma_grid_valid[order(arma_grid_valid$AIC), ]
arma_grid_bic <- arma_grid_valid[order(arma_grid_valid$BIC), ]

cat("\nTop 10 ARMA models by AIC:\n")
print(head(arma_grid_aic, 10))
cat("\nTop 10 ARMA models by BIC:\n")
print(head(arma_grid_bic, 10))

best_aic_row <- arma_grid_aic[1, ]
best_bic_row <- arma_grid_bic[1, ]

best_aic_index <- which(
  arma_grid$p == best_aic_row$p &
    arma_grid$q == best_aic_row$q &
    arma_grid$include_mean == best_aic_row$include_mean
)[1]

best_bic_index <- which(
  arma_grid$p == best_bic_row$p &
    arma_grid$q == best_bic_row$q &
    arma_grid$include_mean == best_bic_row$include_mean
)[1]

best_aic_fit <- arma_fits[[best_aic_index]]
best_bic_fit <- arma_fits[[best_bic_index]]

cat("\nAIC-selected ARMA model:\n")
print(best_aic_row)
print(best_aic_fit)
cat("\nBIC-selected ARMA model:\n")
print(best_bic_row)
print(best_bic_fit)

write.csv(arma_grid_valid, "ARMA_Grid_Search_Results.csv", row.names = FALSE)

# ------------------------------------------------------------
# 8. Residual analysis for the selected ARMA models
# ------------------------------------------------------------
arma_residual_diagnostics <- function(fit, p, q, model_name) {
  e <- as.numeric(na.omit(residuals(fit)))
  cat("\nResidual diagnostics for", model_name, "\n")
  cat("Ljung-Box test on residuals:\n")
  print(Box.test(e, lag = 20, type = "Ljung-Box", fitdf = p + q))
  cat("Ljung-Box test on squared residuals:\n")
  print(Box.test(e^2, lag = 20, type = "Ljung-Box"))
  cat("ARCH-LM test on residuals:\n")
  print(FinTS::ArchTest(e, lags = 12, demean = TRUE))
}

arma_residual_diagnostics(
  best_aic_fit, best_aic_row$p, best_aic_row$q, "AIC-selected ARMA"
)
arma_residual_diagnostics(
  best_bic_fit, best_bic_row$p, best_bic_row$q, "BIC-selected ARMA"
)

png("ARIMA_AIC_Residual_Diagnostics.png", width = 1200, height = 900, res = 130)
forecast::checkresiduals(best_aic_fit)
dev.off()

png("ARIMA_BIC_Residual_Diagnostics.png", width = 1200, height = 900, res = 130)
forecast::checkresiduals(best_bic_fit)
dev.off()

# ------------------------------------------------------------
# 9. Rolling one-step-ahead ARMA forecasts
# ------------------------------------------------------------
rolling_arma_forecast <- function(fitted_model, training_returns, testing_returns) {
  history <- as.numeric(training_returns)
  predictions <- numeric(length(testing_returns))

  for (i in seq_along(testing_returns)) {
    updated_model <- forecast::Arima(history, model = fitted_model)
    predictions[i] <- as.numeric(forecast::forecast(updated_model, h = 1)$mean[1])
    history <- c(history, testing_returns[i])
  }
  predictions
}

pred_arma_aic <- rolling_arma_forecast(best_aic_fit, train_returns, test_returns)
pred_arma_bic <- rolling_arma_forecast(best_bic_fit, train_returns, test_returns)

pred_price_arma_aic <- previous_prices * exp(pred_arma_aic)
pred_price_arma_bic <- previous_prices * exp(pred_arma_bic)

# ------------------------------------------------------------
# 10. ARMA-GARCH specifications and estimation
# ------------------------------------------------------------
make_garch_spec <- function(p, q, include_mean, distribution) {
  rugarch::ugarchspec(
    variance.model = list(
      model = "sGARCH",
      garchOrder = c(1, 1)
    ),
    mean.model = list(
      armaOrder = c(p, q),
      include.mean = include_mean
    ),
    distribution.model = distribution
  )
}

garch_specs <- list(
  ARMA55_GARCH_Normal = make_garch_spec(5, 5, TRUE, "norm"),
  ARMA55_GARCH_StudentT = make_garch_spec(5, 5, TRUE, "std"),
  ARMA00_GARCH_Normal = make_garch_spec(0, 0, FALSE, "norm"),
  ARMA00_GARCH_StudentT = make_garch_spec(0, 0, FALSE, "std")
)

# Use the complete return series but reserve the last 500 returns
# from estimation. This permits rolling one-step-ahead forecasts.
garch_fits <- list()
for (model_name in names(garch_specs)) {
  cat("\nFitting:", model_name, "\n")
  garch_fits[[model_name]] <- rugarch::ugarchfit(
    spec = garch_specs[[model_name]],
    data = all_returns,
    out.sample = test_size,
    solver = "hybrid"
  )
  print(garch_fits[[model_name]])
}

# Information criteria for the four GARCH specifications.
garch_ic <- do.call(rbind, lapply(names(garch_fits), function(model_name) {
  fit <- garch_fits[[model_name]]
  ic <- as.numeric(rugarch::infocriteria(fit))
  data.frame(
    Model = model_name,
    LogLikelihood = as.numeric(rugarch::likelihood(fit)),
    AIC = ic[1],
    BIC = ic[2]
  )
}))

cat("\nARMA-GARCH information criteria:\n")
print(garch_ic)
write.csv(garch_ic, "GARCH_Information_Criteria.csv", row.names = FALSE)

# Additional standardised-residual diagnostics.
garch_residual_diagnostics <- function(fit, p, q, model_name) {
  z <- as.numeric(na.omit(residuals(fit, standardize = TRUE)))
  cat("\nStandardised-residual diagnostics for", model_name, "\n")
  cat("Ljung-Box on standardised residuals:\n")
  print(Box.test(z, lag = 20, type = "Ljung-Box", fitdf = p + q))
  cat("Ljung-Box on squared standardised residuals:\n")
  print(Box.test(z^2, lag = 20, type = "Ljung-Box"))
  cat("ARCH-LM on standardised residuals:\n")
  print(FinTS::ArchTest(z, lags = 12, demean = TRUE))
}

garch_residual_diagnostics(garch_fits$ARMA55_GARCH_Normal, 5, 5, "ARMA(5,5)-GARCH Normal")
garch_residual_diagnostics(garch_fits$ARMA55_GARCH_StudentT, 5, 5, "ARMA(5,5)-GARCH Student-t")
garch_residual_diagnostics(garch_fits$ARMA00_GARCH_Normal, 0, 0, "ARMA(0,0)-GARCH Normal")
garch_residual_diagnostics(garch_fits$ARMA00_GARCH_StudentT, 0, 0, "ARMA(0,0)-GARCH Student-t")

# Full rugarch print output also reports weighted Ljung-Box,
# weighted ARCH-LM, Nyblom stability, sign-bias and Pearson GOF tests.
# The printed model summaries above contain these diagnostics.

# ------------------------------------------------------------
# 11. Rolling one-step-ahead GARCH forecasts
# ------------------------------------------------------------
garch_forecasts <- lapply(garch_fits, function(fit) {
  rugarch::ugarchforecast(
    fit,
    n.ahead = 1,
    n.roll = test_size - 1
  )
})

garch_pred_returns <- lapply(garch_forecasts, function(fc) {
  as.numeric(fitted(fc))
})

garch_pred_volatility <- lapply(garch_forecasts, function(fc) {
  as.numeric(sigma(fc))
})

for (model_name in names(garch_pred_returns)) {
  stopifnot(length(garch_pred_returns[[model_name]]) == test_size)
  stopifnot(length(garch_pred_volatility[[model_name]]) == test_size)
}

garch_pred_prices <- lapply(garch_pred_returns, function(pred) {
  previous_prices * exp(pred)
})

# ------------------------------------------------------------
# 12. Machine-learning dataset: five lagged returns
# ------------------------------------------------------------
# embed() orders the current target first, followed by its lags.
lag_matrix <- embed(all_returns, 6)
data_ml <- as.data.frame(lag_matrix)
colnames(data_ml) <- c("Target", "Lag1", "Lag2", "Lag3", "Lag4", "Lag5")

data_ml <- data_ml[complete.cases(data_ml), ]
if (any(!is.finite(as.matrix(data_ml)))) {
  stop("Non-finite values detected in the machine-learning dataset.")
}

test_ml_size <- 500
n_ml <- nrow(data_ml)
if (n_ml <= test_ml_size + 100) {
  stop("Insufficient observations for ML training and testing.")
}

train_ml <- data_ml[1:(n_ml - test_ml_size), ]
test_ml <- data_ml[(n_ml - test_ml_size + 1):n_ml, ]

features <- c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5")
X_train <- train_ml[, features]
y_train <- train_ml$Target
X_test <- test_ml[, features]
y_test <- test_ml$Target

cat("\nMachine-learning dataset dimensions:\n")
print(dim(data_ml))
cat("Training dimensions:\n")
print(dim(train_ml))
cat("Testing dimensions:\n")
print(dim(test_ml))
print(colSums(is.na(data_ml)))

# Chronological 80/20 split within the training dataset.
n_ml_train <- nrow(train_ml)
n_valid <- floor(0.20 * n_ml_train)
fit_end <- n_ml_train - n_valid

ml_fit <- train_ml[1:fit_end, ]
ml_valid <- train_ml[(fit_end + 1):n_ml_train, ]

# ------------------------------------------------------------
# 13. Random Forest: grid search and final model
# ------------------------------------------------------------
rf_grid <- expand.grid(
  mtry = 1:5,
  nodesize = c(1, 5, 10)
)
rf_grid$RMSE <- NA_real_

for (i in seq_len(nrow(rf_grid))) {
  set.seed(123)
  rf_candidate <- randomForest::randomForest(
    Target ~ .,
    data = ml_fit,
    ntree = 500,
    mtry = rf_grid$mtry[i],
    nodesize = rf_grid$nodesize[i]
  )
  pred_valid_rf <- predict(rf_candidate, newdata = ml_valid)
  rf_grid$RMSE[i] <- sqrt(mean((ml_valid$Target - pred_valid_rf)^2))
}

rf_grid <- rf_grid[order(rf_grid$RMSE), ]
cat("\nRandom Forest grid-search results:\n")
print(rf_grid)
best_rf <- rf_grid[1, ]
cat("\nSelected Random Forest configuration:\n")
print(best_rf)
write.csv(rf_grid, "Random_Forest_Grid_Search.csv", row.names = FALSE)

set.seed(123)
rf_model <- randomForest::randomForest(
  Target ~ .,
  data = train_ml,
  ntree = 500,
  mtry = best_rf$mtry,
  nodesize = best_rf$nodesize,
  importance = TRUE
)
print(rf_model)

pred_rf <- as.numeric(predict(rf_model, newdata = test_ml))
pred_price_rf <- previous_prices * exp(pred_rf)

cat("\nRandom Forest variable importance:\n")
print(randomForest::importance(rf_model))
png("NIFTY50_RF_Variable_Importance.png", width = 1200, height = 800, res = 130)
randomForest::varImpPlot(rf_model)
dev.off()

# ------------------------------------------------------------
# 14. XGBoost: expanded grid search and round selection
# ------------------------------------------------------------
dtrain <- xgboost::xgb.DMatrix(
  data = as.matrix(ml_fit[, features]),
  label = ml_fit$Target
)
dvalid <- xgboost::xgb.DMatrix(
  data = as.matrix(ml_valid[, features]),
  label = ml_valid$Target
)

xgb_grid <- expand.grid(
  max_depth = c(1, 2, 3, 4, 5, 7, 10),
  eta = c(0.03, 0.05),
  min_child_weight = c(1, 5, 10),
  nrounds = c(50, 100, 200)
)
xgb_grid$RMSE <- NA_real_

for (i in seq_len(nrow(xgb_grid))) {
  params <- list(
    objective = "reg:squarederror",
    max_depth = xgb_grid$max_depth[i],
    eta = xgb_grid$eta[i],
    min_child_weight = xgb_grid$min_child_weight[i],
    subsample = 0.8,
    colsample_bytree = 0.8
  )

  set.seed(123)
  xgb_candidate <- xgboost::xgb.train(
    params = params,
    data = dtrain,
    nrounds = xgb_grid$nrounds[i],
    verbose = 0
  )

  pred_valid_xgb <- predict(xgb_candidate, dvalid)
  xgb_grid$RMSE[i] <- sqrt(mean((ml_valid$Target - pred_valid_xgb)^2))
}

xgb_grid <- xgb_grid[order(xgb_grid$RMSE), ]
cat("\nTop 20 XGBoost grid-search configurations:\n")
print(head(xgb_grid, 20))
write.csv(xgb_grid, "XGBoost_Grid_Search.csv", row.names = FALSE)

# Refine the number of rounds for the best depth/eta/child-weight
# combination. The reported experiment selected 5 rounds, the
# second-best result in the 1:10 round search, after comparing it
# with the marginally lower RMSE at 1 round.
best_xgb_base <- xgb_grid[1, ]
round_params <- list(
  objective = "reg:squarederror",
  max_depth = best_xgb_base$max_depth,
  eta = best_xgb_base$eta,
  min_child_weight = best_xgb_base$min_child_weight,
  subsample = 0.8,
  colsample_bytree = 0.8
)

round_grid <- data.frame(nrounds = 1:10, RMSE = NA_real_)
for (i in seq_len(nrow(round_grid))) {
  set.seed(123)
  xgb_candidate <- xgboost::xgb.train(
    params = round_params,
    data = dtrain,
    nrounds = round_grid$nrounds[i],
    verbose = 0
  )
  pred_valid_xgb <- predict(xgb_candidate, dvalid)
  round_grid$RMSE[i] <- sqrt(mean((ml_valid$Target - pred_valid_xgb)^2))
}

round_grid <- round_grid[order(round_grid$RMSE), ]
cat("\nXGBoost boosting-round refinement:\n")
print(round_grid)

# Retain the five-round configuration used in the reported experiment.
chosen_xgb_rounds <- 5L
cat("Selected XGBoost base configuration:\n")
print(best_xgb_base)
cat("Selected number of boosting rounds:", chosen_xgb_rounds, "\n")

# ------------------------------------------------------------
# 15. Fit final XGBoost model and generate test predictions
# ------------------------------------------------------------
dtrain_full <- xgboost::xgb.DMatrix(
  data = as.matrix(train_ml[, features]),
  label = train_ml$Target
)
dtest <- xgboost::xgb.DMatrix(
  data = as.matrix(test_ml[, features])
)

set.seed(123)
xgb_model <- xgboost::xgb.train(
  params = round_params,
  data = dtrain_full,
  nrounds = chosen_xgb_rounds,
  verbose = 0
)
print(xgb_model)

pred_xgb <- as.numeric(predict(xgb_model, dtest))
pred_price_xgb <- previous_prices * exp(pred_xgb)

cat("\nXGBoost feature importance:\n")
xgb_importance <- xgboost::xgb.importance(
  feature_names = features,
  model = xgb_model
)
print(xgb_importance)
png("NIFTY50_XGBoost_Variable_Importance.png", width = 1200, height = 800, res = 130)
xgboost::xgb.plot.importance(xgb_importance)
dev.off()

# ------------------------------------------------------------
# 16. Forecast accuracy metrics and consolidated comparison
# ------------------------------------------------------------
calculate_metrics <- function(actual, predicted) {
  error <- as.numeric(actual) - as.numeric(predicted)
  c(
    ME = mean(error),
    MAE = mean(abs(error)),
    RMSE = sqrt(mean(error^2)),
    MAPE = mean(abs(error / as.numeric(actual))) * 100
  )
}

forecast_metrics <- rbind(
  ARIMA_AIC = calculate_metrics(actual_test_prices, pred_price_arma_aic),
  ARIMA_BIC = calculate_metrics(actual_test_prices, pred_price_arma_bic),
  ARMA55_GARCH_Normal = calculate_metrics(actual_test_prices, garch_pred_prices$ARMA55_GARCH_Normal),
  ARMA55_GARCH_StudentT = calculate_metrics(actual_test_prices, garch_pred_prices$ARMA55_GARCH_StudentT),
  ARMA00_GARCH_Normal = calculate_metrics(actual_test_prices, garch_pred_prices$ARMA00_GARCH_Normal),
  ARMA00_GARCH_StudentT = calculate_metrics(actual_test_prices, garch_pred_prices$ARMA00_GARCH_StudentT),
  Tuned_Random_Forest = calculate_metrics(actual_test_prices, pred_price_rf),
  Tuned_XGBoost = calculate_metrics(actual_test_prices, pred_price_xgb)
)

cat("\nConsolidated out-of-sample price-level metrics:\n")
print(round(forecast_metrics, 6))
write.csv(as.data.frame(forecast_metrics), "NIFTY50_Forecast_Metrics.csv")

# ------------------------------------------------------------
# 17. Save forecast values and dates
# ------------------------------------------------------------
prediction_table <- data.frame(
  Date = test_dates,
  Actual_Closing_Price = actual_test_prices,
  ARIMA_AIC = pred_price_arma_aic,
  ARIMA_BIC = pred_price_arma_bic,
  ARMA55_GARCH_Normal = garch_pred_prices$ARMA55_GARCH_Normal,
  ARMA55_GARCH_StudentT = garch_pred_prices$ARMA55_GARCH_StudentT,
  ARMA00_GARCH_Normal = garch_pred_prices$ARMA00_GARCH_Normal,
  ARMA00_GARCH_StudentT = garch_pred_prices$ARMA00_GARCH_StudentT,
  Tuned_Random_Forest = pred_price_rf,
  Tuned_XGBoost = pred_price_xgb
)
write.csv(prediction_table, "NIFTY50_Test_Predictions.csv", row.names = FALSE)

# ------------------------------------------------------------
# 18. Actual-versus-predicted plots
# ------------------------------------------------------------
plot_theme <- theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    plot.subtitle = element_text(size = 10),
    axis.title = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# ARIMA: actual black, AIC model blue, BIC model orange.
arima_plot_data <- data.frame(
  Date = test_dates,
  Actual = actual_test_prices,
  AIC_Model = pred_price_arma_aic,
  BIC_Model = pred_price_arma_bic
)

p_arima <- ggplot(arima_plot_data, aes(Date)) +
  geom_line(aes(y = Actual, colour = "Actual Price"), linewidth = 0.8) +
  geom_line(aes(y = AIC_Model, colour = "ARIMA AIC-selected"), linewidth = 0.7) +
  geom_line(aes(y = BIC_Model, colour = "ARIMA BIC-selected"), linewidth = 0.7) +
  scale_colour_manual(values = c(
    "Actual Price" = "black",
    "ARIMA AIC-selected" = "blue",
    "ARIMA BIC-selected" = "orange"
  )) +
  labs(title = "NIFTY 50: Actual vs Predicted Closing Prices",
       subtitle = "ARIMA Models | Rolling One-Step-Ahead Forecasts (500 Observations)",
       x = "Date", y = "NIFTY 50 Closing Price", colour = "Series") +
  plot_theme

ggplot2::ggsave("NIFTY50_ARIMA_Actual_vs_Predicted.png", p_arima,
                width = 14, height = 7, dpi = 300)

# GARCH: ARMA(5,5)-GARCH Student-t selected; ARMA(0,0)-GARCH Normal benchmark.
garch_plot_data <- data.frame(
  Date = test_dates,
  Actual = actual_test_prices,
  ARMA55_GARCH_StudentT = garch_pred_prices$ARMA55_GARCH_StudentT,
  ARMA00_GARCH_Normal = garch_pred_prices$ARMA00_GARCH_Normal
)

p_garch <- ggplot(garch_plot_data, aes(Date)) +
  geom_line(aes(y = Actual, colour = "Actual Price"), linewidth = 0.8) +
  geom_line(aes(y = ARMA55_GARCH_StudentT, colour = "ARIMA(5,1,5)-GARCH Student-t"), linewidth = 0.7) +
  geom_line(aes(y = ARMA00_GARCH_Normal, colour = "ARIMA(0,1,0)-GARCH Normal"), linewidth = 0.7) +
  scale_colour_manual(values = c(
    "Actual Price" = "black",
    "ARIMA(5,1,5)-GARCH Student-t" = "blue",
    "ARIMA(0,1,0)-GARCH Normal" = "orange"
  )) +
  labs(title = "NIFTY 50: Actual vs Predicted Closing Prices",
       subtitle = "ARMA-GARCH Models | Rolling One-Step-Ahead Forecasts (500 Observations)",
       x = "Date", y = "NIFTY 50 Closing Price", colour = "Series") +
  plot_theme

ggplot2::ggsave("NIFTY50_GARCH_Actual_vs_Predicted.png", p_garch,
                width = 14, height = 7, dpi = 300)

# Random Forest: actual black, tuned RF orange.
rf_plot_data <- data.frame(
  Date = test_dates,
  Actual = actual_test_prices,
  Tuned_RF = pred_price_rf
)

p_rf <- ggplot(rf_plot_data, aes(Date)) +
  geom_line(aes(y = Actual, colour = "Actual Price"), linewidth = 0.8) +
  geom_line(aes(y = Tuned_RF, colour = "Tuned Random Forest"), linewidth = 0.7) +
  scale_colour_manual(values = c(
    "Actual Price" = "black",
    "Tuned Random Forest" = "orange"
  )) +
  labs(title = "NIFTY 50: Actual vs Predicted Closing Prices",
       subtitle = "Tuned Random Forest | Rolling One-Step-Ahead Forecasts (500 Observations)",
       x = "Date", y = "NIFTY 50 Closing Price", colour = "Series") +
  plot_theme

ggplot2::ggsave("NIFTY50_Tuned_RF_Actual_vs_Predicted.png", p_rf,
                width = 14, height = 7, dpi = 300)

# XGBoost: actual black, tuned XGBoost orange.
xgb_plot_data <- data.frame(
  Date = test_dates,
  Actual = actual_test_prices,
  Tuned_XGBoost = pred_price_xgb
)

p_xgb <- ggplot(xgb_plot_data, aes(Date)) +
  geom_line(aes(y = Actual, colour = "Actual Price"), linewidth = 0.8) +
  geom_line(aes(y = Tuned_XGBoost, colour = "Tuned XGBoost"), linewidth = 0.7) +
  scale_colour_manual(values = c(
    "Actual Price" = "black",
    "Tuned XGBoost" = "orange"
  )) +
  labs(title = "NIFTY 50: Actual vs Predicted Closing Prices",
       subtitle = "Tuned XGBoost | Rolling One-Step-Ahead Forecasts (500 Observations)",
       x = "Date", y = "NIFTY 50 Closing Price", colour = "Series") +
  plot_theme

ggplot2::ggsave("NIFTY50_Tuned_XGBoost_Actual_vs_Predicted.png", p_xgb,
                width = 14, height = 7, dpi = 300)

# Combined comparison of representative models.
combined_plot_data <- data.frame(
  Date = test_dates,
  Actual = actual_test_prices,
  ARIMA_BIC = pred_price_arma_bic,
  GARCH_55_StudentT = garch_pred_prices$ARMA55_GARCH_StudentT,
  Random_Forest = pred_price_rf,
  XGBoost = pred_price_xgb
)

combined_long <- reshape(
  combined_plot_data,
  varying = c("Actual", "ARIMA_BIC", "GARCH_55_StudentT", "Random_Forest", "XGBoost"),
  v.names = "Price",
  timevar = "Model",
  times = c("Actual Price", "ARIMA(0,1,0)", "ARMA(5,5)-GARCH Student-t", "Tuned Random Forest", "Tuned XGBoost"),
  idvar = "Date",
  direction = "long"
)

p_all <- ggplot(combined_long, aes(Date, Price, colour = Model)) +
  geom_line(linewidth = 0.65) +
  scale_colour_manual(values = c(
    "Actual Price" = "black",
    "ARIMA(0,1,0)" = "blue",
    "ARMA(5,5)-GARCH Student-t" = "red",
    "Tuned Random Forest" = "orange",
    "Tuned XGBoost" = "darkgreen"
  )) +
  labs(title = "NIFTY 50: Actual vs Predicted Closing Prices",
       subtitle = "Representative Models | 500-Observation Test Set",
       x = "Date", y = "NIFTY 50 Closing Price", colour = "Model") +
  plot_theme

ggplot2::ggsave("NIFTY50_All_Models_Comparison.png", p_all,
                width = 14, height = 7, dpi = 300)

# Plot conditional volatility forecasts for the four GARCH models.
volatility_data <- data.frame(
  Date = test_dates,
  ARMA55_Normal = garch_pred_volatility$ARMA55_GARCH_Normal,
  ARMA55_StudentT = garch_pred_volatility$ARMA55_GARCH_StudentT,
  ARMA00_Normal = garch_pred_volatility$ARMA00_GARCH_Normal,
  ARMA00_StudentT = garch_pred_volatility$ARMA00_GARCH_StudentT
)

volatility_long <- reshape(
  volatility_data,
  varying = names(volatility_data)[-1],
  v.names = "Conditional_SD",
  timevar = "Model",
  times = names(volatility_data)[-1],
  idvar = "Date",
  direction = "long"
)

p_volatility <- ggplot(volatility_long, aes(Date, Conditional_SD, colour = Model)) +
  geom_line(linewidth = 0.65) +
  labs(title = "GARCH Conditional Volatility Forecasts",
       x = "Date", y = "Conditional Standard Deviation", colour = "Model") +
  plot_theme

ggplot2::ggsave("NIFTY50_GARCH_Conditional_Volatility.png", p_volatility,
                width = 14, height = 7, dpi = 300)

# ------------------------------------------------------------
# 19. Save model-selection and experiment summaries
# ------------------------------------------------------------
selected_arma <- data.frame(
  Criterion = c("AIC", "BIC"),
  p = c(best_aic_row$p, best_bic_row$p),
  q = c(best_aic_row$q, best_bic_row$q),
  Include_Mean = c(best_aic_row$include_mean, best_bic_row$include_mean),
  AIC = c(best_aic_row$AIC, best_bic_row$AIC),
  BIC = c(best_aic_row$BIC, best_bic_row$BIC)
)
write.csv(selected_arma, "Selected_ARMA_Models.csv", row.names = FALSE)

selected_ml <- data.frame(
  Model = c("Random Forest", "XGBoost"),
  RF_mtry = c(best_rf$mtry, NA),
  RF_nodesize = c(best_rf$nodesize, NA),
  RF_ntree = c(500, NA),
  XGB_max_depth = c(NA, best_xgb_base$max_depth),
  XGB_eta = c(NA, best_xgb_base$eta),
  XGB_min_child_weight = c(NA, best_xgb_base$min_child_weight),
  XGB_nrounds = c(NA, chosen_xgb_rounds)
)
write.csv(selected_ml, "Selected_ML_Configurations.csv", row.names = FALSE)

cat("\n============================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("============================================================\n")
cat("Generated plots, CSV tables and forecasts in:\n")
cat(normalizePath(getwd()), "\n")
cat("Key files:\n")
cat("  NIFTY50_ARIMA_Actual_vs_Predicted.png\n")
cat("  NIFTY50_GARCH_Actual_vs_Predicted.png\n")
cat("  NIFTY50_Tuned_RF_Actual_vs_Predicted.png\n")
cat("  NIFTY50_Tuned_XGBoost_Actual_vs_Predicted.png\n")
cat("  NIFTY50_All_Models_Comparison.png\n")
cat("  NIFTY50_Forecast_Metrics.csv\n")