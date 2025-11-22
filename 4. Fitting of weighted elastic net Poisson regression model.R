# Load required libraries
library(tidyverse)
library(glmnet)
library(boot)
library(pROC)

# Set working directory
setwd("./data")

# Read training data with weights
train <- read.csv("train_weights.csv")

# Convert categorical variables to factors
train$Smoke_status <- as.factor(train$Smoke_status)

# Extract clinical meaningful predictors and treatment variable
X <- data.matrix(train[, c(8:14, 23, 26, 27, 31)])
X_all <- data.matrix(train[, 8:34]) 
treated <- train[, 7]

# Create interaction matrix
interaction_matrix <- X * treated

# Combine interaction terms with original variables
X_P <- cbind(treated, X_all, interaction_matrix)

# Name interaction terms
colnames(interaction_matrix) <- paste0(colnames(X), "_treated")
colnames(X_P) <- c("treated", colnames(X_all), colnames(interaction_matrix))

# Response variable and sample weights
Y_P <- train$Dementia
sample_weights <- train$sample_weights

# Elastic Net Poisson regression with treated penalty set to 0
set.seed()  
penalty_factors <- rep(1, ncol(X_P))
penalty_factors[1] <- 0

# Cross-validation to select optimal lambda
cvfit <- cv.glmnet(X_P, Y_P, family = "poisson", alpha = 0.10, 
                   penalty.factor = penalty_factors, weights = sample_weights)

# Get optimal lambda and fit final model
best_lambda <- cvfit$lambda.min
Poisson_fit <- glmnet(X_P, Y_P, family = "poisson", alpha = 0.10, 
                      lambda = best_lambda, penalty.factor = penalty_factors, 
                      weights = sample_weights)

# Print model coefficients
print(coef(Poisson_fit))

# Extract variable names and create formula string
coef_names <- rownames(coef(Poisson_fit))
non_zero_coef <- coef(Poisson_fit)[, 1] != 0
selected_vars <- coef_names[non_zero_coef]

# Create formula string with "+" separator
formula_string <- paste(selected_vars, collapse = " + ")
cat("Formula from Elastic Net:\n", formula_string, "\n\n")


## Remove interaction terms without main effects based on formula_string
formula <- Dementia ~ treated + ... + ...:treated + ...:treated


# Fit final Poisson regression model
final_model <- glm(formula, data = train, family = poisson, weights = train$sample_weights)
summary(final_model)

# Generate predicted values
train$predicted_rate <- predict(final_model, type = "response")

# Calculate predicted probability of event occurrence
calculate_event_probability <- function(lambda) {
  1 - dpois(0, lambda)
}

train$pred_prob <- sapply(train$predicted_rate, calculate_event_probability)

# Calculate original AUC
roc_original <- roc(train$Dementia, train$pred_prob)
original_auc <- auc(roc_original)

# Bootstrap function for AUC calculation
calculate_auc_bootstrap <- function(data, indices) {
  bootstrap_sample <- data[indices, ]
  roc_obj <- roc(bootstrap_sample$Dementia, bootstrap_sample$pred_prob, quiet = TRUE)
  return(as.numeric(auc(roc_obj)))
}

# Execute bootstrap (1000 replicates)
set.seed()
boot_results <- boot(data = train, statistic = calculate_auc_bootstrap, R = 1000)

# Calculate confidence intervals using percentile method
boot_ci <- boot.ci(boot_results, type = "perc", conf = 0.95)

# Output results
cat("\n=== AUC Results ===\n")
cat("Original AUC:", round(original_auc, 4), "\n")
cat("Bootstrap AUC mean:", round(mean(boot_results$t), 4), "\n")
cat("Percentile 95% CI: [", round(boot_ci$perc[4], 4), ", ", round(boot_ci$perc[5], 4), "]\n")