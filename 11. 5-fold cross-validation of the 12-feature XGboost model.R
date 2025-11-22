rm(list = ls())
library(xgboost)
library(tidyverse)
library(caret)
library(openxlsx)

setwd("./data")

# Load datasets
train <- read.csv("train_weights.csv")
test <- read.csv("test_data.csv")

# Convert categorical variables to factors
train$Smoke_status <- as.factor(train$Smoke_status)
test$Smoke_status <- as.factor(test$Smoke_status)

# Fit Poisson regression model (from step 9)
formula <- Dementia ~ treated + ... + ...:treated + ...:treated
model <- glm(formula, family = poisson(link = "log"), data = train, weights = train$sample_weights)

# Function to calculate predicted probability of event occurrence
calculate_event_probability <- function(lambda) {
  1 - dpois(0, lambda)
}

# Function to calculate ITE values (from step 9)
calculate_ite <- function(data, model, cols_range) {
  factual_data <- data[, cols_range]
  factual_data$treated <- 1
  factual_data$predicted_rate <- predict(model, newdata = factual_data, type = "response")
  factual_data$pred_prob <- sapply(factual_data$predicted_rate, calculate_event_probability)
  
  counterfactual_data <- data[, cols_range]
  counterfactual_data$treated <- 0
  counterfactual_data$predicted_rate <- predict(model, newdata = counterfactual_data, type = "response")
  counterfactual_data$pred_prob <- sapply(counterfactual_data$predicted_rate, calculate_event_probability)
  
  (factual_data$pred_prob - counterfactual_data$pred_prob) * 100
}

# Calculate ITE for training and test sets (from step 9)
train$Dementia_dif <- calculate_ite(train, model, 7:34)
test$Dementia_dif <- calculate_ite(test, model, 6:33)

# Select important variables for XGBoost (from step 9)
selected_vars <- c("Age", "eGFR", "Alcohol_drink", "High_school", "BMI", "Statin", 
                   "Smoke_status", "SBP", "RxHBP", "Aspirin", "Potassium", "LDL", "Dementia_dif")

# Prepare training and test data (from step 9)
train_xgb <- train %>% select(all_of(selected_vars)) %>% as.data.frame(lapply(., as.numeric))
Xxgboost <- as.matrix(train_xgb[, -13])
label_vector <- train_xgb$Dementia_dif

selected_vars_test <- c("Age", "eGFR", "Alcohol_drink", "High_school", "BMI", "Statin", 
                        "Smoke_status", "SBP", "RxHBP", "Aspirin", "Potassium", "LDL")
test_xgb <- test %>% select(all_of(selected_vars_test)) %>% as.data.frame(lapply(., as.numeric))
Xxgboost_test <- as.matrix(test_xgb)

# Load predefined parameters for 5-fold cross validation (from step 9)
load("./output/dementia_params_12_variables.RData")

# 5-fold cross validation
set.seed()
folds <- createFolds(label_vector, k = 5, list = FALSE)

# Initialize storage
test_predictions_matrix <- matrix(0, nrow = nrow(test), ncol = 5)
fold_metrics <- data.frame(
  fold = 1:5,
  RMSE = NA_real_,
  R_squared = NA_real_
)

for (i in 1:5) {
  # Split data
  test_idx <- which(folds == i)
  train_idx <- setdiff(seq_len(nrow(train)), test_idx)
  
  dtrain <- xgb.DMatrix(data = Xxgboost[train_idx, ], label = label_vector[train_idx])
  dtest <- xgb.DMatrix(data = Xxgboost[test_idx, ])
  
  # Train model
  xgb_model <- xgb.train(params = cv_params, data = dtrain, nrounds = 1000, verbose = 0)
  
  # Calculate training performance
  fold_pred_train <- predict(xgb_model, newdata = dtest)
  actual_values <- label_vector[test_idx]
  
  fold_metrics$RMSE[i] <- sqrt(mean((actual_values - fold_pred_train)^2))
  fold_metrics$R_squared[i] <- cor(actual_values, fold_pred_train)^2
  
  # Predict test set
  test_predictions_matrix[, i] <- predict(xgb_model, newdata = Xxgboost_test)
}

# Process test set predictions
test_predictions_df <- as.data.frame(test_predictions_matrix)
colnames(test_predictions_df) <- paste0("fold_", 1:5, "_pred")
test_predictions_df$Merged_fold <- rowMeans(test_predictions_df)
test_with_predictions <- cbind(test, test_predictions_df)

# Calculate test set performance
test_actual <- test$Dementia_dif
test_performance <- data.frame(
  fold = c(1:5, "Merged_fold"),
  RMSE = c(
    sapply(1:5, function(i) sqrt(mean((test_actual - test_predictions_df[[i]])^2))),
    sqrt(mean((test_actual - test_predictions_df$Merged_fold)^2))
  ),
  R_squared = c(
    sapply(1:5, function(i) cor(test_actual, test_predictions_df[[i]])^2),
    cor(test_actual, test_predictions_df$Merged_fold)^2
  )
)

# Output results
cat("5-Fold Cross Validation Results:\n")
cat("Training set performance:\n")
print(fold_metrics)

cat("\nTest set performance:\n")
print(test_performance)

# Save results
performance_summary <- data.frame(
  Dataset = c(rep("Training", 5), rep("Test", 6)),
  Fold = c(1:5, 1:5, "Merged_fold"),
  RMSE = c(fold_metrics$RMSE, test_performance$RMSE),
  R_squared = c(fold_metrics$R_squared, test_performance$R_squared)
)

results_list <- list(
  "Test_Predictions" = test_with_predictions,
  "Performance_Summary" = performance_summary,
  "Training_Metrics" = fold_metrics,
  "Test_Metrics" = test_performance
)

write.xlsx(results_list, file = "./output/dementia_5fold_cv_results.xlsx")
write.xlsx(test_with_predictions, file = "./output/test_set_predictions.xlsx")