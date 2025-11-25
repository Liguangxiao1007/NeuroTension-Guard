# Load required libraries
library(tidyverse)
library(pROC)
library(ResourceSelection)

# 以下包在代码中没有使用，可以删除：
# library(glmnet)   # 没有使用glmnet相关函数
# library(boot)     # 没有使用boot相关函数

# Set working directory
setwd("./data")

# Read training data
train <- read.csv("train_weights.csv")

# Convert categorical variables to factors
train$Smoke_status <- as.factor(train$Smoke_status)

# Formula based on step 4. Fitting of weighted elastic net Poisson regression model
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

# Calculate predicted probabilities for all data points
train$pred_prob <- sapply(train$predicted_rate, calculate_event_probability)

# Print first few rows of predicted values and probabilities
head(train[, c("predicted_rate", "pred_prob")])

# Hosmer-Lemeshow test on training set
hl_test <- hoslem.test(train$Dementia, train$pred_prob, g = 10)
print(hl_test)

# Training set calibration curve
# Create data frame
data <- data.frame(predicted_risk = train$pred_prob, observed_outcome = train$Dementia)
colnames(data) <- c('predicted_risk', 'observed_outcome')

# Group by deciles of predicted risk
data <- data %>%
  mutate(decile = ntile(predicted_risk, 10))

# Calculate mean predicted probability and actual observed probability for each decile
grouped <- data %>%
  group_by(decile) %>%
  summarise(
    mean_predicted_risk = mean(predicted_risk),
    mean_observed_outcome = mean(observed_outcome),
    n = n(),
    sd_observed_outcome = sd(observed_outcome)
  ) %>%
  mutate(
    se = sd_observed_outcome / sqrt(n),
    lower_ci = mean_observed_outcome - 1.96 * se,
    upper_ci = mean_observed_outcome + 1.96 * se
  )

# Fit linear model and get coefficients and confidence intervals
lm_model <- lm(mean_observed_outcome ~ mean_predicted_risk, data = grouped)
intercept <- coef(lm_model)[1]
slope <- coef(lm_model)[2]

# Get 95% confidence intervals for coefficients
conf_int <- confint(lm_model)
intercept_ci <- conf_int[1, ]
slope_ci <- conf_int[2, ]

# Plot calibration curve
ggplot(grouped, aes(x = mean_predicted_risk, y = mean_observed_outcome)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0, color = "black", linewidth = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "solid", color = "red", linewidth = 0.8) +
  coord_cartesian(xlim = c(0, 0.20), ylim = c(0, 0.20)) +
  labs(x = "Predicted risk of dementia", y = "Observed absolute risk of dementia within deciles") +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 14),
    axis.ticks.length.x = unit(0.1,'cm'), 
    axis.ticks.length.y = unit(0.1,'cm'),
    axis.ticks.x = element_line(colour = "black", size = 1),
    axis.ticks.y = element_line(colour = "black", size = 1),
    panel.border = element_rect(color = "black", fill = NA, size = 1)
  )

# Output regression coefficients
cat("Regression coefficient details:\n")
cat("Intercept:", round(intercept, 4), "\n")
cat("95% CI: [", round(intercept_ci[1], 4), ", ", round(intercept_ci[2], 4), "]\n")
cat("Slope:", round(slope, 4), "\n")
cat("95% CI: [", round(slope_ci[1], 4), ", ", round(slope_ci[2], 4), "]\n")

# Calculate Expected Calibration Error (ECE) - using decile grouping
calculate_ece_decile <- function(actual, predicted_probs, n_bins = 10) {
  # Use same binning method as calibration curve: group by deciles of predicted probability
  deciles <- ntile(predicted_probs, n_bins)
  
  # Calculate statistics for each decile
  ece_data <- data.frame(
    actual = actual,
    predicted = predicted_probs,
    decile = deciles
  ) %>%
    group_by(decile) %>%
    summarise(
      n = n(),
      avg_predicted = mean(predicted),
      avg_actual = mean(actual),
      .groups = 'drop'
    )
  
  # Calculate weighted mean absolute error
  ece <- sum(ece_data$n * abs(ece_data$avg_predicted - ece_data$avg_actual)) / length(actual)
  
  return(list(
    ece = ece, 
    details = ece_data
  ))
}

# Calculate ECE
ece_result_decile <- calculate_ece_decile(train$Dementia, train$pred_prob)

# Enhanced calibration plot showing all metrics
ggplot(ece_result_decile$details, aes(x = avg_predicted, y = avg_actual)) +
  geom_point(aes(size = n), color = "blue", alpha = 0.7) +
  geom_errorbar(aes(ymin = avg_actual - 1.96*sqrt(avg_actual*(1-avg_actual)/n), 
                    ymax = avg_actual + 1.96*sqrt(avg_actual*(1-avg_actual)/n)), 
                width = 0.005, color = "gray", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", size = 0.8) +
  coord_cartesian(xlim = c(0, 0.20), ylim = c(0, 0.20)) +
  labs(
    x = "Average predicted risk", 
    y = "Average observed risk",
    title = paste0("Calibration curve\n",
                   "ECE = ", round(ece_result_decile$ece, 4),
                   ", HL p = ", round(hl_test$p.value, 4)),
    size = "Sample size"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_blank()
  )

# Output final results summary
cat("\n=== Model calibration metrics summary ===\n")
cat("Expected Calibration Error (ECE):", round(ece_result_decile$ece, 5), "\n")
cat("Hosmer-Lemeshow test p-value:", round(hl_test$p.value, 4), "\n")
cat("Calibration slope:", round(slope, 4), "(95% CI:", round(slope_ci[1], 4), "-", round(slope_ci[2], 4), ")\n")
cat("Calibration intercept:", round(intercept, 4), "(95% CI:", round(intercept_ci[1], 4), "-", round(intercept_ci[2], 4), ")\n")