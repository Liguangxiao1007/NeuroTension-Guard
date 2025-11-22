# Load required libraries for distance calculation and visualization
library(kmed)  # For Gower distance calculation
library(umap)   # For UMAP dimensionality reduction
library(ggplot2) # For data visualization

# Set seed for reproducibility
set.seed()

# Load training dataset
data <- read.csv("./data/train_dataset.csv")   

# Select features for analysis (columns 6-32)
X_gower <- data[, 6:32]

# Define numeric variables for Gower distance calculation
g_idnum <- c("Age", "BMI", "NoHBPRx", "SBP", "DBP", "Pulse", "FPG", "eGFR", 
             "TC", "HDL", "TG", "LDL", "Potassium", "Na", "CL")

# Define categorical variables for Gower distance calculation
g_idcat <- c("Female", "Race_black", "Smoke_status", "Alcohol_drink", 
             "Physical_activity", "High_school", "RxHBP", "Aspirin", "Statin", 
             "MI_history", "Angina_history", "HF_history")

# Calculate Gower distance for mixed data types
train_gower <- distmix(X_gower, method = "gower", idnum = g_idnum, idcat = g_idcat)

# Convert distance object to matrix format
train_gower_matrix <- as.matrix(train_gower)

# Configure UMAP parameters: 10 neighbors, minimum distance 0.85
umap_config <- umap.defaults
umap_config$n_neighbors <- 10
umap_config$min_dist <- 0.85

# Apply UMAP dimensionality reduction
umap_result <- umap(train_gower_matrix, config = umap_config)

# Create initial UMAP visualization
umap_df <- as.data.frame(umap_result$layout)
ggplot(umap_df, aes(x = V1, y = V2)) +
  geom_point() +
  labs(title = "UMAP Projection", x = "UMAP1", y = "UMAP2")

# Append UMAP coordinates to original dataset
data$Dim1 <- umap_df$V1
data$Dim2 <- umap_df$V2

# Save dataset with UMAP coordinates
write.csv(data, "./output/train_with_umap.csv", row.names = FALSE)

# Reload data for visualization
Output <- read.csv("./output/train_with_umap.csv")   

# Convert treatment variable to factor for proper plotting
Output$treated <- as.factor(Output$treated)

# Create UMAP visualization colored by treatment group
p1 <- ggplot(Output, aes(x = Dim1, y = Dim2, color = treated)) +
  geom_point(size = 1) + 
  theme_minimal() + 
  scale_color_manual(values = c("#00A5E7", "#E97F5C")) +
  theme(
    legend.position = c(0.01, 0.99), # Position legend in top-left corner
    legend.justification = c(0, 1),  # Align legend to top-left corner
    panel.grid.major = element_blank(), # Remove major grid lines
    panel.grid.minor = element_blank()  # Remove minor grid lines
  ) +
  guides(color = guide_legend(override.aes = list(shape = 15, size = 5))) # Square legend keys

# Display the plot
p1