# Shapley Analysis with Random Forest on Traffic Accident Severity

## Introduction
This project presents a comprehensive analysis of traffic accident severity using the Shapley value analysis in combination with a Random Forest model. Our dataset encompasses various factors contributing to accidents, enabling us to predict and understand the severity of these incidents better.

## Dataset
The dataset, `Accdataset_hk_PS_BAEL_Combined.csv`, includes detailed records of traffic accidents, featuring variables such as accident location, type, causes, road features, and weather conditions. Our target variable is `Accident_Severity_C`, categorized into several severity levels.

## Analysis Overview
We employ Random Forest classifiers and regressors to model the accident severity based on a range of predictors excluding identifiers and remarks. The Shapley value analysis further allows us to interpret the model by quantifying each feature's contribution to the prediction.

### Key Steps:
- Data preprocessing: Fill missing values and select relevant features.
- Model training: Utilize Random Forest for both classification and regression tasks.
- Hyperparameter tuning: Conduct GridSearchCV to find optimal model parameters.
- Shapley value computation: Utilize `shap.TreeExplainer` to calculate Shap values for model interpretation.
- Visualization: Generate plots like waterfall and summary plots to visualize the impact of features.

## Installation
Ensure you have the following Python libraries installed:
- pandas
- numpy
- sklearn
- shap

You can install these packages using pip:
```bash
pip install pandas numpy sklearn shap

