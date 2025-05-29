# Employee Attrition Analysis and Prediction Project

This project analyzes and predicts employee attrition using synthetic HR data, implementing a full ML pipeline with MLflow for experiment tracking and model management. Key components include:

- Comprehensive EDA of 23 employee attributes
- Data preprocessing and feature engineering
- Machine learning model training with hyperparameter tuning
- MLflow integration for experiment tracking and model registry
- Model deployment and performance comparison

## Key Features

### 1. MLflow Integration:

- End-to-end experiment tracking (parameters, metrics, artifacts)
- Model registry for version control and deployment management
- Automatic logging of model performance metrics
- Artifact storage for models and visualizations

![MLFlow Experiments](./MLflow_1.png)
![MLFlow Comparison](./MLflow_2.png)
![MLFlow Metrics](./MLflow_3.png)

### 2. Machine Learning Pipeline:

- Three model architectures: Logistic Regression, XGBoost, Random Forest
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Performance metrics tracking (accuracy, AUC-ROC, F1-score)

### 3. Data Insights:

- 47.5% attrition rate in training data
- Monthly income and company tenure identified as top predictors
- Key attrition drivers: overtime work, low job satisfaction
- Employees with 2-5 years tenure most likely to leave

### 4. Visualization:

- Interactive MLflow UI for experiment comparison
- Confusion matrices and ROC curves
- Feature importance plots

## Key Findings

- XGBoost achieved near-perfect AUC-ROC (1.00) indicating excellent class separation
- Monthly income is the strongest predictor of attrition
- Employees working overtime have 3.2x higher attrition risk
- Optimal hyperparameters differ significantly from default values

## Benefits

- Reproducibility: Full experiment tracking via MLflow
- Model Management: Version control and staging/production pipelines
- Insight Discovery: Identify key attrition drivers and risk factors
- Deployment Ready: Models packaged with dependencies for serving
