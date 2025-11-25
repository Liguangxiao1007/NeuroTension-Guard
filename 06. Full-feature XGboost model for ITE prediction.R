rm(list = ls())
library(glmnet)
library(xgboost)
library(shapviz)
library(tidyverse)
library(pROC)
library(openxlsx)
#library(devtools) 
#install_github("xnie/rlearner") ###if Rlearner not installed
library(rlearner)

setwd("./data")

# Load training dataset
train <- read.csv("train_weights.csv")

# Convert categorical variables to factors
train$Smoke_status <- as.factor(train$Smoke_status)

# Fit Poisson regression model
formula <- Dementia ~ treated + ... + ...:treated + ...:treated
model <- glm(formula, family = poisson(link = "log"), data = train, weights = train$sample_weights)
summary(model)

# Function to calculate predicted probability of event occurrence
calculate_event_probability <- function(lambda) {
  1 - dpois(0, lambda)
}

# Calculate ITE values for training set
calculate_ite <- function(data, model) {
  # Factual scenario: all treated
  factual_data <- data[, 7:34]
  factual_data$treated <- 1
  factual_data$predicted_rate <- predict(model, newdata = factual_data, type = "response")
  factual_data$pred_prob <- sapply(factual_data$predicted_rate, calculate_event_probability)
  
  # Counterfactual scenario: all untreated
  counterfactual_data <- data[, 7:34]
  counterfactual_data$treated <- 0
  counterfactual_data$predicted_rate <- predict(model, newdata = counterfactual_data, type = "response")
  counterfactual_data$pred_prob <- sapply(counterfactual_data$predicted_rate, calculate_event_probability)
  
  # Calculate absolute risk difference
  (factual_data$pred_prob - counterfactual_data$pred_prob) * 100
}

train$Dementia_dif <- calculate_ite(train, model)

# Prepare data for XGBoost
train <- train[, -c(1, 2, 4, 5, 35:38)]

train_xgb <- cbind(train[, -c(1:3)])
train_xgb <- as.data.frame(lapply(train_xgb, as.numeric))
Xxgboost <- as.matrix(train_xgb[, -28])
label_vector <- train_xgb$Dementia_dif
data_dmatrix <- xgb.DMatrix(data = Xxgboost, label = label_vector)

# XGBoost modeling
set.seed()
cvxboost <- cvboost(Xxgboost, train_xgb$Dementia_dif, objective = "reg:squarederror")
cv_params <- cvxboost$best_param
xgb_model <- xgb.train(data = data_dmatrix, cv_params, nrounds = 1000)
train$pre_dementia_dif <- predict(xgb_model, newdata = Xxgboost)

# Save best model parameters for next step 5-fold cross validation
save(cv_params, file = "./output/all_dementia.RData")

# Save data file
write.xlsx(train, file = "./output/train_pred_all.xlsx")

# Variable importance and SHAP analysis
importance_matrix <- xgb.importance(feature_names = colnames(Xxgboost), model = xgb_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

Dementia_shap <- shapviz(xgb_model, X_pred = Xxgboost)

# Enhanced SHAP importance plot
shap_plot <- sv_importance(Dementia_shap, kind = "beeswarm", max_display = 30, show_numbers = TRUE) +
  scale_colour_gradient(low = "#0073E6", high = "#fb0155", breaks = c(0, 1), labels = c("Low", "High")) +
  theme_bw() +
  theme(
    panel.grid = element_blank(), 
    legend.position = c(0.9, 0.20),
    plot.margin = unit(c(0.3, 0.1, 0.1, 1.1), "inches"),
    panel.spacing = unit(0, "lines"),
    axis.text = element_text(size = 12),
    text = element_text(size = 12)
  ) +
  geom_hline(yintercept = 0.10, linetype = "dashed", color = "#D9232D")

print(shap_plot)