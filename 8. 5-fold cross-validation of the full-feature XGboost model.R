rm(list = ls())
library(xgboost)
library(tidyverse)
library(caret)
library(openxlsx)

setwd("./data")

# Load training data
train <- read.csv("train_weights.csv")

# Convert categorical variables to factors
train$Smoke_status <- as.factor(train$Smoke_status)

# Fit Poisson regression model
formula <- Dementia ~ treated + ... + ...:treated + ...:treated
model <- glm(formula, family = poisson(link = "log"), data = train, weights = train$sample_weights)

# Function to calculate predicted probability of event occurrence
calculate_event_probability <- function(lambda) {
  1 - dpois(0, lambda)
}

# Calculate ITE values for training set
factual_data <- train[, 7:34]
factual_data$treated <- 1
factual_data$predicted_rate <- predict(model, newdata = factual_data, type = "response")
factual_data$pred_prob <- sapply(factual_data$predicted_rate, calculate_event_probability)

counterfactual_data <- train[, 7:34]
counterfactual_data$treated <- 0
counterfactual_data$predicted_rate <- predict(model, newdata = counterfactual_data, type = "response")
counterfactual_data$pred_prob <- sapply(counterfactual_data$predicted_rate, calculate_event_probability)

# Calculate absolute risk difference
train$Dementia_dif <- (factual_data$pred_prob - counterfactual_data$pred_prob) * 100
train <- train[, -c(1, 2, 4, 5, 35:38)]

# Prepare data for XGBoost
train_xgb <- cbind(train[, -c(1:3)])
train_xgb <- as.data.frame(lapply(train_xgb, as.numeric))
Xxgboost <- as.matrix(train_xgb[, -28])
label_vector <- train_xgb$Dementia_dif

# Load predefined best cv_params from step6
load("./output/all_dementia 92.RData")

# 5-fold cross validation with random sampling
set.seed()
folds <- createFolds(label_vector, k = 5, list = FALSE, returnTrain = FALSE)

# Initialize storage
train_predictions <- numeric(nrow(train))
fold_metrics <- data.frame(
  fold = 1:5,
  RMSE = NA_real_,
  R_squared = NA_real_
)

fold_predictions_list <- list()

for (i in 1:5) {
  # Split data
  test_idx <- which(folds == i)
  train_idx <- setdiff(seq_len(nrow(train)), test_idx)
  
  dtrain <- xgb.DMatrix(data = Xxgboost[train_idx, ], label = label_vector[train_idx])
  dtest <- xgb.DMatrix(data = Xxgboost[test_idx, ])
  
  # Train model with predefined cv_params
  xgb_model <- xgb.train(
    params = cv_params,
    data = dtrain,
    nrounds = 1000,
    verbose = 0
  )
  
  # Save predictions
  fold_pred <- predict(xgb_model, newdata = dtest)
  train_predictions[test_idx] <- fold_pred
  
  # Calculate performance metrics
  actual_values <- label_vector[test_idx]
  pred_values <- fold_pred
  
  RMSE_fold <- sqrt(mean((actual_values - pred_values)^2))
  R_squared_fold <- cor(actual_values, pred_values)^2
  
  fold_metrics$RMSE[i] <- RMSE_fold
  fold_metrics$R_squared[i] <- R_squared_fold
  
  # Save detailed predictions
  fold_predictions_list[[i]] <- data.frame(
    row_index = test_idx,
    fold = i,
    actual = actual_values,
    predicted = pred_values
  )
}

# Overall performance evaluation
train$Dementia_dif_pred <- train_predictions
train$fold <- folds

RMSE_overall <- sqrt(mean((train$Dementia_dif - train$Dementia_dif_pred)^2))
R_squared_overall <- cor(train$Dementia_dif, train$Dementia_dif_pred)^2

# Combined evaluation across all folds
all_fold_predictions <- do.call(rbind, fold_predictions_list)
RMSE_combined <- sqrt(mean((all_fold_predictions$actual - all_fold_predictions$predicted)^2))
R_squared_combined <- cor(all_fold_predictions$actual, all_fold_predictions$predicted)^2

# Output results
cat("5-Fold Cross Validation Results:\n")
cat("Overall RMSE:", round(RMSE_overall, 4), "\n")
cat("Overall R-squared:", round(R_squared_overall, 4), "\n")
cat("Combined RMSE:", round(RMSE_combined, 4), "\n")
cat("Combined R-squared:", round(R_squared_combined, 4), "\n")

# Save results
fold_predictions_all <- do.call(rbind, fold_predictions_list)
performance_summary <- data.frame(
  Evaluation_Method = c("Point_Estimate", "Fold_Combined"),
  RMSE = c(RMSE_overall, RMSE_combined),
  R_squared = c(R_squared_overall, R_squared_combined)
)

results_list <- list(
  "Training_Predictions" = train,
  "Fold_Predictions" = fold_predictions_all,
  "Performance_Summary" = performance_summary,
  "Fold_Metrics" = fold_metrics
)

write.xlsx(results_list, file = "./output/training_5fold_cv_results.xlsx")
write.xlsx(train, file = "./output/training_with_fold_predictions.xlsx")