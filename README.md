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

## 🏗️ Code Structure
### XGBoost Implementation Pipeline
```
code/
├── 01_data_preprocessing/
│   └── 1. Data imputation.txt
│
├── 02_phenomapping/
│   ├── 2. Phenomapping using Gower's distance.txt
│   └── 3. Determination of sample weights.txt
│
├── 03_ite_estimation/
│   ├── 4. Fitting of weighted elastic net Poisson regression model.txt
│   └── 5. Calibration of the Poisson regression model.txt
│
├── 04_xgboost_modeling/
│   ├── 6. Full-feature XGboost model for ITE prediction.txt
│   ├── 7. Calibration of the full-feature XGboost model.txt
│   ├── 8. 5-fold cross-validation of the full-feature XGboost model.txt
│   ├── 9. 12-feature XGboost model for ITE prediction.txt
│   ├── 10. Calibration of the 12-feature XGboost model.txt
│   └── 11. 5-fold cross-validation of the 12-feature XGboost model.txt
│
└── 05_model_discrimination/
    └── 12. Calculation of C-for-Benefit.txt
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
Expected calibration error: 0.034
Calibration slope (95%CI): 1.005 (0.962, 1.047)
Calibration intercept: -0.002 (-0.051, 0.047)
C-for-benefit: 0.517 (0.494, 0.540)

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
@article{guo2024phenomaps,
  title={Phenomaps-Optimized Personalized Intensive Blood Pressure Management Using Machine Learning for Typical Patients: A New Frontier in Dementia Prevention for Individuals Without a Stroke History},
  author={Guo, Xiaofan and Sun, Guozhe and Zhong, Shanshan and others},
  journal={Under Review},
  year={2025}
}

## 📞 Contact

Prof. Yingxian Sun: yxsun@cmu.edu.cn

Prof. Chuansheng Zhao: cszhao@cmu.edu.cn