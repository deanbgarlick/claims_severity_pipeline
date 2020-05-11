# claims_severity_pipeline

A simple data science pipeline for predicting motor insurance claim amounts applied to the Allstate Claims Severity Kaggle Dataset https://www.kaggle.com/c/allstate-claims-severity.

This pipeline makes use of:
 - Bayesian feature encoding for categorical variables using the category-encoders pypi package
 - XGBoost's Gradient Boosted Machines
 - Facebook Research's open source Gaussian Process based optimisation libraries Ax/Botorch
 - DataBrick's open source MlFlow for tracking and logging pipeline optimistion and prediction