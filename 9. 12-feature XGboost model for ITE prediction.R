rm(list = ls())
library(xgboost)
library(tidyverse)
library(openxlsx)

setwd("./data")

# Load datasets
train <- read.csv("train_weights.csv")
test <- read.csv("test_data.csv")

# Convert categorical variables to factors
train$Smoke_status <- as.factor(train$Smoke_status)
test$Smoke_status <- as.factor(test$Smoke_status)

# Fit Poisson regression model
formula <- Dementia ~ treated + ... + ...:treated + ...:treated
model <- glm(formula, family = poisson(link = "log"), data = train, weights = train$sample_weights)

# Function to calculate predicted probability of event occurrence
calculate_event_probability <- function(lambda) {
  1 - dpois(0, lambda)
}

# Function to calculate ITE values
calculate_ite <- function(data, model, cols_range) {
  # Factual scenario: all treated
  factual_data <- data[, cols_range]
  factual_data$treated <- 1
  factual_data$predicted_rate <- predict(model, newdata = factual_data, type = "response")
  factual_data$pred_prob <- sapply(factual_data$predicted_rate, calculate_event_probability)
  
  # Counterfactual scenario: all untreated
  counterfactual_data <- data[, cols_range]
  counterfactual_data$treated <- 0
  counterfactual_data$predicted_rate <- predict(model, newdata = counterfactual_data, type = "response")
  counterfactual_data$pred_prob <- sapply(counterfactual_data$predicted_rate, calculate_event_probability)
  
  # Calculate absolute risk difference
  (factual_data$pred_prob - counterfactual_data$pred_prob) * 100
}

# Calculate ITE for training and test sets
train$Dementia_dif <- calculate_ite(train, model, 7:34)
test$Dementia_dif <- calculate_ite(test, model, 6:33)

# Select important variables for XGBoost
selected_vars <- c("Age", "eGFR", "Alcohol_drink", "High_school", "BMI", "Statin", 
                   "Smoke_status", "SBP", "RxHBP", "Aspirin", "Potassium", "LDL", "Dementia_dif")

# Prepare training data
train_xgb <- train %>% select(all_of(selected_vars))
train_xgb <- as.data.frame(lapply(train_xgb, as.numeric))
Xxgboost <- as.matrix(train_xgb[, -13])
label_vector <- train_xgb$Dementia_dif
data_dmatrix <- xgb.DMatrix(data = Xxgboost, label = label_vector)

# XGBoost modeling with selected variables
set.seed()
cvxboost <- cvboost(Xxgboost, train_xgb$Dementia_dif, objective = "reg:squarederror")
cv_params <- cvxboost$best_param

# Save model parameters for subsequent 5-fold cross validation
save(cv_params, file = "./output/dementia_params 12 variables.RData")

xgb_model <- xgb.train(data = data_dmatrix, cv_params, nrounds = 1000)

# Generate predictions for training set
train$pre_dementia_dif <- predict(xgb_model, newdata = Xxgboost)

# Generate predictions for test set
selected_vars_test <- c("Age", "eGFR", "Alcohol_drink", "High_school", "BMI", "Statin", 
                        "Smoke_status", "SBP", "RxHBP", "Aspirin", "Potassium", "LDL")

test_xgb <- test %>% select(all_of(selected_vars_test))
test_xgb <- as.data.frame(lapply(test_xgb, as.numeric))
Xxgboost_test <- as.matrix(test_xgb)
test$pre_dementia_dif <- predict(xgb_model, newdata = Xxgboost_test)

# Save results
write.xlsx(train, file = "./output/train_pred_selected_vars.xlsx")
write.xlsx(test, file = "./output/test_pred_selected_vars.xlsx")