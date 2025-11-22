# Load required libraries
library(openxlsx)
library(DataExplorer)
library(missRanger)

# Read data from Excel file
# Note: Using generic path and filename for privacy
setwd("./data")  # Generic working directory
data <- read.xlsx('dataset.xlsx', sheet = 1)

# Check missing values pattern
plot_missing(data)

# Display dataset structure
colnames(data)
dim(data)

# Convert specified variables to factor type
# These are categorical variables that should be treated as factors
vars_to_factor <- c("treated", "Female", "Race_black", "Smoke_status", 
                    "Alcohol_drink", "Physical_activity", "High_school",
                    "RxHBP", "Aspirin", "Statin", "MI_history", 
                    "Angina_history", "HF_history")

for (var in vars_to_factor) {
  data[[var]] <- as.factor(data[[var]])
}

# Split data into training and test sets based on Train_test indicator
# Train_test = 1: Training set, Train_test = 2: Test set
train <- data[data$Train_test == 1, ]
test <- data[data$Train_test == 2, ]

# Check dimensions of split datasets
dim(train)
dim(test)

# Visualize missing values in both sets
plot_missing(train)
plot_missing(test)

# Impute missing values using missRanger (Random Forest based imputation)
# Set seed for reproducibility (seed number not specified for privacy)
set.seed()  # Seed set for reproducibility
train <- missRanger(train)

set.seed()  # Seed set for reproducibility
test <- missRanger(test)

# Save imputed datasets with generic file paths
write.csv(train, "./output/train_imputed.csv")
write.csv(test, "./output/test_imputed.csv")

# Verify no missing values remain after imputation
plot_missing(train)
plot_missing(test)