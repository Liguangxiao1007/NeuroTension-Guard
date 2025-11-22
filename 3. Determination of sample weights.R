# Load required libraries - 实际使用的包
library(tidyverse)  # 用于数据处理和ggplot2
library(kmed)       # 用于Gower距离计算
library(sigmoid)    # 用于sigmoid函数

# 以下包在代码中没有使用，可以删除：
# library(openxlsx)   # 没有读取Excel文件，使用的是read.csv
# library(boot)       # 没有使用boot相关函数
# library(caret)      # 没有使用caret相关函数
# library(pROC)       # 没有使用ROC相关函数
# library(ResourceSelection) # 没有使用该包函数
# library(survival)   # 没有使用生存分析函数

# Set working directory
setwd("./data")

# Read training data
train <- read.csv("train_dataset.csv")

# Convert categorical variables to factors
train$Smoke_status <- as.factor(train$Smoke_status)

# Select features for Gower distance calculation
X_gower <- train[, 7:33]

# Define numeric and categorical variables
g_idnum <- c("Age", "BMI", "NoHBPRx", "SBP", "DBP", "Pulse", "FPG", "eGFR", 
             "TC", "HDL", "TG", "LDL", "Potassium", "Na", "CL")
g_idcat <- c("Female", "Race_black", "Smoke_status", "Alcohol_drink", 
             "Physical_activity", "High_school", "RxHBP", "Aspirin", "Statin", 
             "MI_history", "Angina_history", "HF_history")

# Calculate Gower distance
train_gower <- distmix(X_gower, method = "gower", idnum = g_idnum, idcat = g_idcat)

# Find optimal reference patient using centroid method
gower_matrix <- as.matrix(train_gower)
diag(gower_matrix) <- 0
avg_distances <- rowMeans(gower_matrix)
best_index <- which.min(avg_distances)

cat("Optimal reference patient: Sample", best_index, "\n")
cat("Average Gower distance:", round(avg_distances[best_index], 4), "\n")

# Get distances from optimal reference patient
index_gower <- as.matrix(train_gower)[best_index, ]

# Define exponential values to test
exp_values <- c(2, 3, 5, 10, 15, 20, 50)
model_metrics <- numeric(length(exp_values))

# Define helper functions
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  exp_x / sum(exp_x)
}

# Weight transformation function
weight_transform <- function(x) {
  sigmoid(x)
}

# Define model formula
formula <- Dementia ~ treated + Age + Female + Race_black + BMI + Smoke_status + 
  Physical_activity + High_school + RxHBP + NoHBPRx + Aspirin + Statin + 
  MI_history + Angina_history + HF_history + SBP + DBP + Pulse + FPG +  
  eGFR + TC + HDL + TG + LDL + Potassium + Na + CL

n_samples <- length(index_gower)

# Test different exponential values
for (i in seq_along(exp_values)) {
  exp_value <- exp_values[i]
  
  # Calculate sample weights using modified method
  similarity_metric <- (1 - index_gower)^exp_value
  similarity_softmax <- softmax(similarity_metric)
  scaled_similarity_softmax <- similarity_softmax * n_samples
  sample_weights <- weight_transform(scaled_similarity_softmax)
  
  # Fit Poisson regression model
  poisson_model <- glm(formula, data = train, family = poisson, weights = sample_weights)
  model_metrics[i] <- AIC(poisson_model)
  
  cat("exp =", exp_value, 
      "| Similarity range:[", round(min(similarity_metric), 4), ",", round(max(similarity_metric), 4), "]",
      "| Final weights range:[", round(min(sample_weights), 6), ",", round(max(sample_weights), 6), "]",
      "| AIC:", round(model_metrics[i], 2), "\n")
}

# Find optimal exponential value
best_exp <- exp_values[which.min(model_metrics)]
cat("Optimal exp value:", best_exp, "with AIC:", round(min(model_metrics), 2), "\n")

# Calculate final weights using optimal exp value
similarity_metric_best <- (1 - index_gower)^best_exp
similarity_softmax_best <- softmax(similarity_metric_best)
scaled_similarity_softmax_best <- similarity_softmax_best * n_samples
train$sample_weights <- weight_transform(scaled_similarity_softmax_best)

# Validate final weights
cat("\n=== Final Weights Statistics ===\n")
cat("Weight sum:", round(sum(train$sample_weights), 6), "\n")
cat("Average weight:", round(mean(train$sample_weights), 6), "\n")
cat("Weight range: [", round(min(train$sample_weights), 6), ",", round(max(train$sample_weights), 6), "]\n")

# Save results
write.csv(train, file = "./output/train_weights_gower.csv")

# Plot AIC vs exp values
plot_df <- data.frame(exp = exp_values, AIC = model_metrics)
ggplot(plot_df, aes(x = exp, y = AIC)) +
  geom_line() +
  geom_point() +
  geom_point(data = plot_df[which.min(plot_df$AIC), ], color = "red", size = 3) +
  labs(title = "AIC vs Exponential Value", x = "Exponential Value", y = "AIC") +
  theme_minimal()

cat("\nAnalysis completed! Results saved to file.\n")