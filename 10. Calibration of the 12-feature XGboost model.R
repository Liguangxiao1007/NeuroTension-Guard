rm(list = ls())
library(openxlsx)
library(ggplot2)
library(tidyverse)

#### Test Set Calibration Curve ####
data_test <- read.xlsx('./data/test_pred_selected_vars.xlsx', sheet = 1)
observed_dif <- data_test$Dementia_dif
predicted_dif <- data_test$pre_dementia_dif

# Create data frame
data <- data.frame(predicted_dif, observed_dif)

# Group by deciles of predicted difference
data <- data %>%
  mutate(decile = ntile(predicted_dif, 10))

# Calculate mean predicted difference and actual observed difference for each decile
grouped <- data %>%
  group_by(decile) %>%
  summarise(
    mean_predicted_dif = mean(predicted_dif),
    mean_observed_dif = mean(observed_dif),
    n = n(),
    sd_observed_dif = sd(observed_dif)
  ) %>%
  mutate(
    se = sd_observed_dif / sqrt(n),
    lower_ci = mean_observed_dif - 1.96 * se,
    upper_ci = mean_observed_dif + 1.96 * se
  )

# Linear model
lm_model <- lm(mean_observed_dif ~ mean_predicted_dif, data = grouped)
intercept <- coef(lm_model)[1]
slope <- coef(lm_model)[2]

# Get 95% confidence intervals for coefficients
conf_int <- confint(lm_model)
intercept_ci <- conf_int[1, ]
slope_ci <- conf_int[2, ]

### Calculate ATE ECE (core calibration metric)
calculate_ate_ece <- function(actual_ate, predicted_ate, n_bins = 10) {
  # Use decile grouping
  deciles <- ntile(predicted_ate, n_bins)
  
  # Calculate statistics for each decile
  ece_data <- data.frame(
    actual_ate = actual_ate,
    predicted_ate = predicted_ate,
    decile = deciles
  ) %>%
    group_by(decile) %>%
    summarise(
      n = n(),
      avg_predicted_ate = mean(predicted_ate),
      avg_actual_ate = mean(actual_ate),
      .groups = 'drop'
    )
  
  # Calculate weighted mean absolute error
  ece <- sum(ece_data$n * abs(ece_data$avg_predicted_ate - ece_data$avg_actual_ate)) / length(actual_ate)
  
  return(list(
    ece = ece, 
    details = ece_data
  ))
}

# Calculate ECE for ATE
ate_ece_result <- calculate_ate_ece(observed_dif, predicted_dif)
cat("ATE Expected Calibration Error (ECE):", round(ate_ece_result$ece, 5), "\n")

# Print detailed bin information
cat("ATE ECE bin details:\n")
print(ate_ece_result$details)

### Plot calibration curve
ggplot(grouped, aes(x = mean_predicted_dif, y = mean_observed_dif)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.01, color = "red") +
  geom_abline(slope = 1, intercept = 0, linetype = "solid", color = "red") +
  coord_cartesian(xlim = c(-2.0, 0), ylim = c(-2.0, 0)) +
  labs(x = "Overall Predicted ARD, %", y = "Observed ARD, %") +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    axis.ticks = element_line(size = 1),
    panel.border = element_rect(color = "black", fill = NA, size = 1)
  )

### Output results summary
cat("\n=== ATE Prediction Calibration Metrics Summary ===\n")
cat("Expected Calibration Error (ECE):", round(ate_ece_result$ece, 5), "\n")
cat("Calibration slope:", round(slope, 4), "(95% CI:", round(slope_ci[1], 4), "-", round(slope_ci[2], 4), ")\n")
cat("Calibration intercept:", round(intercept, 4), "(95% CI:", round(intercept_ci[1], 4), "-", round(intercept_ci[2], 4), ")\n")