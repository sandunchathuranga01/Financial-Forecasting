# Load necessary libraries
if (!require("neuralnet")) install.packages("neuralnet")
if (!require("Metrics")) install.packages("Metrics")
if (!require("readxl")) install.packages("readxl")

library(neuralnet)
library(Metrics)
library(readxl)

# Data Loading
ExchangeData <- read_excel("ExchangeUSD.xlsx")
colnames(ExchangeData)[colnames(ExchangeData) == "USD/EUR"] <- "USD_EUR" # Rename column 

# Create lagged features t-1 to t-4
exchangeRates <- ExchangeData$USD_EUR
ExchangeData$Lag_1 <- c(NA, exchangeRates[1:(length(exchangeRates) - 1)])
ExchangeData$Lag_2 <- c(NA, NA, exchangeRates[1:(length(exchangeRates) - 2)])
ExchangeData$Lag_3 <- c(NA, NA, NA, exchangeRates[1:(length(exchangeRates) - 3)])
ExchangeData$Lag_4 <- c(NA, NA, NA, NA, exchangeRates[1:(length(exchangeRates) - 4)])

# Normalization function (Min-Max Scaling)
normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Apply normalization to numeric columns
NColumns <- sapply(ExchangeData, is.numeric)
ExchangeData[NColumns] <- lapply(ExchangeData[NColumns], normalize)

# Data partitioning
TrainData <- na.omit(ExchangeData[1:400,])
TestData <- na.omit(ExchangeData[401:500,])

# Set seed for reproducibility
set.seed(123)

# Define the formula
formula <- USD_EUR ~ Lag_1 + Lag_2 + Lag_3 + Lag_4

############################  MLP models #######################################

#MLP Neural Network: 1st case (1 hidden layer with 5 neurons)
MLP1 <- neuralnet(formula, data = TrainData, hidden = 5, linear.output = TRUE)
plot(MLP1)

#MLP Neural Network: 2st case (2 hidden layers (5,3))
MLP2 <- neuralnet(formula, data = TrainData, hidden = c(5,3), linear.output = TRUE)
plot(MLP2)

#MLP Neural Network: 3st case (3 hidden layers (5,3,2))
MLP3 <- neuralnet(formula, data = TrainData, hidden = c(5,3,2), linear.output = TRUE)
plot(MLP3)

#MLP Neural Network: 4st case (4 hidden layers (5,4,3,2))
MLP4 <- neuralnet(formula, data = TrainData, hidden = c(5,4,3,2), linear.output = TRUE)
plot(MLP4)


# Prepare test data for predictions
TestingData <- TestData[, c("Lag_1", "Lag_2", "Lag_3", "Lag_4")]
actual <- TestData$USD_EUR

#######################Predictions and Metrics##################################

# Model 1 
predictions1 <- neuralnet::compute(MLP1, TestingData)$net.result
RMSE1  <- rmse(actual, predictions1)
MAE1   <- mae(actual, predictions1)
MAPE1  <- mape(actual, predictions1)
SMAPE1 <- smape(actual, predictions1)
cat(  "\nMLP 01 - RMSE:", RMSE1, "MAE:", MAE1, "MAPE:", MAPE1, "sMAPE:", SMAPE1, "\n")
summary(predictions1)

# Model 2 
predictions2 <- neuralnet::compute(MLP2, TestingData)$net.result
RMSE2  <- rmse(actual, predictions2)
MAE2   <- mae(actual, predictions2)
MAPE2  <- mape(actual, predictions2)
SMAPE2 <- smape(actual, predictions2)
cat("\nMLP 2 - RMSE:", RMSE2, "MAE:", MAE2, "MAPE:", MAPE2, "sMAPE:", SMAPE2, "\n")
summary(predictions2)

# Model 3 
predictions3 <- neuralnet::compute(MLP3, TestingData)$net.result
RMSE3  <- rmse(actual, predictions3)
MAE3   <- mae(actual, predictions3)
MAPE3  <- mape(actual, predictions3)
SMAPE3 <- smape(actual, predictions3)
cat("\nMLP 3 - RMSE:", RMSE3, "MAE:", MAE3, "MAPE:", MAPE3, "sMAPE:", SMAPE3, "\n")
summary(predictions3)

# Model 4 
predictions4 <- neuralnet::compute(MLP4, TestingData)$net.result
RMSE4  <- rmse(actual, predictions4)
MAE4   <- mae(actual, predictions4)
MAPE4  <- mape(actual, predictions4)
SMAPE4 <- smape(actual, predictions4)
cat("\nMLP 4 - RMSE:", RMSE4, "MAE:", MAE4, "MAPE:", MAPE4, "sMAPE:", SMAPE4, "\n")
summary(predictions4)


############################Plot for model######################################

#Model 1
plot(actual, type = 'l', col = 'blue', ylim = range(c(actual, predictions1)), ylab = "Exchange Rate", xlab = "Days", main = "MLP 1 - Actual vs Predicted")
lines(predictions1, col = 'red')
legend("bottomright", col = c("blue", "red"), legend = c("Actual Data", "Predicted Data"), lty = 1)

# Model 2
plot(actual, type = 'l', col = 'blue', ylim = range(c(actual, predictions2)), ylab = "Exchange Rate", xlab = "Days", main = "MLP 2 - Actual vs Predicted")
lines(predictions2, col = 'red')
legend("bottomright", col = c("blue", "red"), legend = c("Actual Data", "Predicted Data"), lty = 1)

# Model 3
plot(actual, type = 'l', col = 'blue', ylim = range(c(actual, predictions3)), ylab = "Exchange Rate", xlab = "Days", main = "MLP 3 - Actual vs Predicted")
lines(predictions3, col = 'red')
legend("bottomright", col = c("blue", "red"), legend = c("Actual Data", "Predicted Data"), lty = 1)

# Model 4
plot(actual, type = 'l', col = 'blue', ylim = range(c(actual, predictions4)), ylab = "Exchange Rate", xlab = "Days", main = "MLP 4 - Actual vs Predicted")
lines(predictions4, col = 'red')
legend("bottomright", col = c("blue", "red"), legend = c("Actual Data", "Predicted Data"), lty = 1)

