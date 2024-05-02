"""
# Author = Cory Cascalheira
# Date = 05/02/2024

The purpose of this script is to execute several ML models to compare performance after PCA, then
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
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import truncnorm, randint
from sklearn.decomposition import PCA

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

# region

# Initialize PCA and keep 95% of the variance
pca = PCA(n_components=0.95)

# Fit the PCA model
pca.fit(cmips_x)

# How many components?
print(pca.n_components_)

# Transform the feature space
cmips_x = pca.transform(cmips_x)

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

# region HYPERPARAMETER TUNING - RANDOM SEARCH - RANDOM FOREST - Lifetime Stressor Count

# Create the parameter search space
param_space = {
    # Randomly sample estimators
    'n_estimators': randint(100, 1000),

    # Randomly sample numbers
    'max_depth': randint(10, 100),

    # Normally distributed max_features, with mean .50 stddev 0.15, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.50, scale=0.15)
}

# Instantiate the model
ml_model = RandomForestRegressor()

# Create the random search algorithm
random_search_rf = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='r2'
)

# Train the random search algorithm
model_rf = random_search_rf.fit(cmips_x, cmips_stressct)

# Save training results to file
with open(my_path + '/doc/random_search_output_regression.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('LIFETIME STRESSOR COUNT - RANDOM SEARCH - PCA - RANDOM FOREST', file=f)
    print('\nBest Parameters', file=f)
    print(model_rf.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_rf.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_rf.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=131, max_depth=48, max_features=0.5919590666199903)

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

# region HYPERPARAMETER TUNING - RANDOM SEARCH - RANDOM FOREST - Lifetime Stressor Severity

# Create the parameter search space
param_space = {
    # Randomly sample estimators
    'n_estimators': randint(100, 1000),

    # Randomly sample numbers
    'max_depth': randint(10, 100),

    # Normally distributed max_features, with mean .50 stddev 0.15, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.50, scale=0.15)
}

# Instantiate the model
ml_model = RandomForestRegressor()

# Create the random search algorithm
random_search_rf = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='r2'
)

# Train the random search algorithm
model_rf = random_search_rf.fit(cmips_x, cmips_stressth)

# Save training results to file
with open(my_path + '/doc/random_search_output_regression.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('LIFETIME STRESSOR SEVERITY - RANDOM SEARCH - PCA - RANDOM FOREST', file=f)
    print('\nBest Parameters', file=f)
    print(model_rf.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_rf.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_rf.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=224, max_depth=21, max_features=0.5141393871864869)

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
