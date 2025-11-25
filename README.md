# Phenomaps-Optimized Personalized Intensive Blood Pressure Management for Dementia Prevention

This repository contains the implementation code for the machine learning framework developed in the study:  
**"Phenomaps-Optimized Personalized Intensive Blood Pressure Management Using Machine Learning for Typical Patients: A New Frontier in Dementia Prevention for Individuals Without a Stroke History"**

## 📋 Study Overview

This study developed machine learning models to predict individualized treatment effects (ITEs) of intensive versus standard blood pressure control. Five algorithms—XGBoost, Transformer, Residual Network (ResNet), Random Forest, and Support Vector Machine (SVM)—were evaluated, with the primary analysis based on the XGBoost algorithm. The analysis includes:
- **Primary outcome**: Dementia risk reduction
- **Secondary outcome**: Increase of adverse events (AEs) of interest

For the secondary outcome (AEs), the analysis directly utilized the same 12 clinical features and weighting scheme identified in the primary dementia outcome analysis, implementing only the 12-feature XGBoost model without developing a full-feature version.

### Key Features
- **XGBoost-Based Primary Analysis**: Main findings and tool implementation based on XGBoost
- **Primary Outcome Focus**: Comprehensive modeling for dementia risk reduction
- **Secondary Outcome Efficiency**: AE analysis builds upon primary outcome findings
- **Phenomap Optimization**: Incorporates Gower's distance-based phenotypic similarity weighting

## 📋 System Requirements

### Software Dependencies and Operating Systems

**R Environment (Required for primary analysis):**
- R version 4.4.0 or higher
- RStudio (recommended) version 2024.04.0 or higher

**Python Environment (Required for secondary analysis):**
- Python version 3.12.2 (primary)

**Operating Systems Tested:**
- Windows 10 (64-bit)

### Required R Packages:
```r
library(openxlsx)      
library(DataExplorer) 
library(missRanger)   
library(kmed)          
library(umap)          
library(ggplot2)       
library(tidyverse)    
library(glmnet)        
library(boot)          
library(pROC)          
library(xgboost)       
library(shapviz)    
.. (other packages as used in the scripts) 
```
### Required Python Packages:

```
random
numpy
pandas
multiprocessing
.. (other packages as used in the scripts) 
```

## 🛠️ Installation Guide

1. Install R and RStudio:
   
  #### Download R from: https://cran.r-project.org/

  #### Download RStudio from: https://posit.co/download/rstudio-desktop/

2. Install Required R Packages:
``` 
packages <- c("openxlsx", "DataExplorer", "missRanger", "kmed", "umap", 
             "ggplot2", "tidyverse", "glmnet", "boot", "pROC", "xgboost", 
             "shapviz","devtools")
install.packages(packages)

library(devtools) 
install_github("xnie/rlearner")
``` 
3. Install Python and Required Packages:
   
  #### Install Python from: https://www.python.org/downloads/

  #### Install required Python packages using pip

``` 
pip install numpy pandas random multiprocessing
``` 
**Typical Installation Time: 20-40 minutes on a normal desktop computer**

## 🚀 Demo Instructions

### Demo Files:

- `Demo_2000_imputation.xlsx` - 2,000 records
- `train_sampled_1000.csv` - 1,000 training records  
- `test_sampled_2000.csv` - 2,000 test records

### Steps:
1. Run data imputation code with `Demo_2000_imputation.xlsx`
2. Run training code with `train_sampled_1000.csv`
3. Run testing code with `test_sampled_2000.csv`

Follow code comments for step-by-step execution.

**Expected Output:** As described by each script name

**Expected Runtime:** 5-10 minutes per script

## 🏗️ Code Structure

### XGBoost Implementation Pipeline
```
code/
├── 01_data_preprocessing/
│   └── 01. Data imputation.R
│
├── 02_phenomapping/
│   ├── 02. Phenomapping using Gower's distance.R
│   └── 03. Determination of sample weights.R
│
├── 03_ite_estimation/
│   ├── 04. Fitting of weighted elastic net Poisson regression model.R
│   └── 05. Calibration of the Poisson regression model.R
│
├── 04_xgboost_modeling/
│   ├── 06. Full-feature XGboost model for ITE prediction.R
│   ├── 07. Calibration of the full-feature XGboost model.R
│   ├── 08. 5-fold cross-validation of the full-feature XGboost model.R
│   ├── 09. 12-feature XGboost model for ITE prediction.R
│   ├── 10. Calibration of the 12-feature XGboost model.R
│   └── 11. 5-fold cross-validation of the 12-feature XGboost model.R
│
└── 05_model_discrimination/
    └── 12. Calculation of C-for-Benefit.py
```

### Model Selection and Primary Analysis
Five machine learning methods were evaluated (XGBoost, Transformer, ResNet, Random Forest, SVM). XGBoost demonstrated the best performance and was selected as the primary algorithm for developing the ITE prediction model and the subsequent online clinical tool.

### Primary Outcome (Dementia) Modeling
The XGBoost model for dementia outcomes was developed through comprehensive analysis:

```
# Optimal hyperparameters identified for full-feature XGboost model
xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.5,
    'colsample_bytree': 0.8,
    'min_child_weight': 9,
    'gamma': 0.13,
    'max_delta_step': 4,
    'objective': 'reg:squarederror'
}
```
Feature Selection and Secondary Outcome Application
Based on SHAP analysis of the primary outcome, 12 key clinical features were identified and subsequently applied to both outcomes:

Age

Estimated glomerular filtration rate

Alcohol drinking

High school

Body Mass Index

Systolic blood pressure

Statin use

Smoke status

Use of antihypertensive medications

Aspirin use

Serum potassium

Low-Density Lipoprotein Cholesterol

**Model Development Approach**

Primary Outcome (Dementia): Full analytical pipeline including feature selection, weighting, and both full-feature and 12-feature XGBoost models

Secondary Outcome (AEs): Direct application of the 12-feature XGBoost model using weights and features derived from primary outcome analysis

## 📊 Model Performance of 12-feature XGboost Model in CRHCP Cohort

**Primary Outcome Evaluation**

R-squared:0.953

Root mean square error:0.177

Calibration slope (95%CI): 1.005 (0.962, 1.047)

Calibration intercept: -0.002 (-0.051, 0.047)

Expected calibration error: 0.034

C-for-benefit: 0.519 (0.494, 0.540)

**Clinical Impact: Observed Treatment Benefits in CRHCP Cohort**

High benefit group: 1.21% absolute risk reduction in dementia outcomes

Moderate benefit group: 0.87% absolute risk reduction

Low benefit group: 0.40% absolute risk reduction

## 📁 Data Availability

**Derivation Cohort**
SPRINT Trial: Access via NHLBI BioLINCC: https://biolincc.nhlbi.nih.gov/

**Validation Cohort**
CRHCP Trial: Available upon reasonable request from Prof. Yingxian Sun

## 📄 Citation

bibtex
@article{
  title={Phenomaps-Optimized Personalized Intensive Blood Pressure Management Using Machine Learning for Typical Patients: A New Frontier in Dementia Prevention for Individuals Without a Stroke History},
  author={Xiaofan Guo, Guozhe Sun, and Shanshan Zhong, Nanxiang Ouyang, Guangxiao Li and others},
  journal={Under Review},
  year={2025}
}

## 📞 Contact

Prof. Yingxian Sun: yxsun@cmu.edu.cn

Prof. Chuansheng Zhao: cszhao@cmu.edu.cn