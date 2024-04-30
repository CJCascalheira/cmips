"""
# Author = Cory Cascalheira
# Date = 04/29/2024

The purpose of this script is to execute several ML models to compare performance, then
performing hyperparameter tuning with random search on the best performing model.

This script regresses:
- Lifetime stressor count
- Lifetime stressor severity

RESOURCES
- Scikit Learn documentation
"""

# Load dependencies
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

# Set working directory
my_path = os.getcwd()

# Set the seed
random.seed(10)

# region PREPARE DATA AND METRICS

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_reg_strain.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'StressTH', 'StressCT'], axis=1)

# Get the two labels
cmips_stressth = cmips_df['StressTH']
cmips_stressct = cmips_df['StressCT']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_stressth.shape)
print(cmips_stressct.shape)

# Transform to matrices
cmips_x = cmips_x.values
cmips_stressth = cmips_stressth.values
cmips_stressct = cmips_stressct.values

# Instantiate the standard scaler
sc = StandardScaler()

# Standardize the feature matrix
cmips_x = sc.fit_transform(cmips_x)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x = cmips_x.values

# Initialize k-fold cross-validation
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Specify the metrics to use
my_metrics = {'explained_variance', 'r2'}

# endregion

#######################################################################################################################

# region LINEAR REGRESSION - Lifetime Stress Severity

# Specify the hyperparameters of the linear regression
linear_reg = LinearRegression()

# Fit the linear regression with k-fold cross-validation
scores_linear_reg = cross_validate(estimator=linear_reg, X=cmips_x, y=cmips_stressth,
                                   scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_explained_variance']), np.std(scores_linear_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_r2']), np.std(scores_linear_reg['test_r2'])))

# endregion

# region ELASTIC NET - Lifetime Stress Severity

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_stressth,
                                    scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_explained_variance']), np.std(scores_elastic_net['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_r2']), np.std(scores_elastic_net['test_r2'])))

# endregion

# region LASSO REGRESSION - Perceived Stress

# Specify the hyperparameters of the lasso regression
lasso_reg = Lasso(alpha=0.5, max_iter=1000)

# Fit the lasso regression with k-fold cross-validation
scores_lasso_reg = cross_validate(estimator=lasso_reg, X=cmips_x, y=cmips_stressth,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_explained_variance']), np.std(scores_lasso_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_r2']), np.std(scores_lasso_reg['test_r2'])))

# endregion

# region RIDGE REGRESSION - Perceived Stress

# Specify the hyperparameters of the ridge regression
ridge_reg = Ridge(alpha=0.5, max_iter=1000)

# Fit the ridge regression with k-fold cross-validation
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_stressth,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

# region RANDOM FOREST - Lifetime Stress Severity

# Specify the hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=1.0)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_stressth,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

# region XGBOOST - Lifetime Stress Severity

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=1000, eta=0.1, max_depth=10, subsample=1.0)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_stressth,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

#######################################################################################################################

# region LINEAR REGRESSION - Lifetime Stress Count

# Specify the hyperparameters of the linear regression
linear_reg = LinearRegression()

# Fit the linear regression with k-fold cross-validation
scores_linear_reg = cross_validate(estimator=linear_reg, X=cmips_x, y=cmips_stressct,
                                   scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_explained_variance']), np.std(scores_linear_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_r2']), np.std(scores_linear_reg['test_r2'])))

# endregion

# region ELASTIC NET - Lifetime Stress Count

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_stressct,
                                    scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_explained_variance']), np.std(scores_elastic_net['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_r2']), np.std(scores_elastic_net['test_r2'])))

# endregion

# region LASSO REGRESSION - Perceived Stress

# Specify the hyperparameters of the lasso regression
lasso_reg = Lasso(alpha=0.5, max_iter=1000)

# Fit the lasso regression with k-fold cross-validation
scores_lasso_reg = cross_validate(estimator=lasso_reg, X=cmips_x, y=cmips_stressct,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_explained_variance']), np.std(scores_lasso_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_r2']), np.std(scores_lasso_reg['test_r2'])))

# endregion

# region RIDGE REGRESSION - Perceived Stress

# Specify the hyperparameters of the ridge regression
ridge_reg = Ridge(alpha=0.5, max_iter=1000)

# Fit the ridge regression with k-fold cross-validation
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_stressct,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

# region RANDOM FOREST - Lifetime Stress Count

# Specify the hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=1.0)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_stressct,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

# region XGBOOST - Lifetime Stress Count

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=1000, eta=0.1, max_depth=10, subsample=1.0)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_stressct,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

#######################################################################################################################

# Because performance was poor for all regressors, random search withheld until after feature selection.
