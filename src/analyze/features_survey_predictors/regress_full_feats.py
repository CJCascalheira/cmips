"""
# Author = Cory J. Cascalheira
# Date = 05/04/2024

The purpose of this script is to add survey-based predictors to models using the reduced feature space.
"""

# region LOAD PACKAGES AND SET METRICS

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

# Set working directory
my_path = os.getcwd()

# Set the seed
random.seed(10)

# Initialize k-fold cross-validation
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Specify the metrics to use
my_metrics = {'explained_variance', 'r2'}

# endregion

# region PREPARE DATA AND METRICS

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_reg_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'PSS_total', 'LEC_total', 'DHEQ_mean', 'OI_mean', 'SOER_total', 'IHS_mean'], axis=1)

# Get the two labels
cmips_dheq = cmips_df['DHEQ_mean']
cmips_oi = cmips_df['OI_mean']
cmips_pss = cmips_df['PSS_total']
cmips_lec = cmips_df['LEC_total']
cmips_ihs = cmips_df['IHS_mean']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_dheq.shape)
print(cmips_oi.shape)
print(cmips_pss.shape)
print(cmips_lec.shape)
print(cmips_ihs.shape)

# Transform to matrices
cmips_x = cmips_x.values
cmips_dheq = cmips_dheq.values
cmips_oi = cmips_oi.values
cmips_pss = cmips_pss.values
cmips_lec = cmips_lec.values
cmips_ihs = cmips_ihs.values

# Standardize the feature matrix
sc = StandardScaler()
cmips_x = sc.fit_transform(cmips_x)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x = cmips_x.values

# endregion

#######################################################################################################################

# region LINEAR REGRESSION - Perceived Stress

# Specify the hyperparameters of the linear regression
linear_reg = LinearRegression()

# Fit the linear regression with k-fold cross-validation
scores_linear_reg = cross_validate(estimator=linear_reg, X=cmips_x, y=cmips_pss,
                                   scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_explained_variance']), np.std(scores_linear_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_r2']), np.std(scores_linear_reg['test_r2'])))

# endregion

# region ELASTIC NET - Perceived Stress

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_pss,
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
scores_lasso_reg = cross_validate(estimator=lasso_reg, X=cmips_x, y=cmips_pss,
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
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_pss,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

# region RANDOM FOREST - Perceived Stress

# Specify the hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=1.0)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_pss,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

# region XGBOOST - Perceived Stress

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=1000, eta=0.1, max_depth=10, subsample=1.0)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_pss,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - XGBOOST - Perceived Stress

# Create the parameter search space
param_space = {
    # Randomly sample estimators
    'n_estimators': randint(100, 1500),

    # Randomly sample numbers
    'max_depth': randint(10, 100),

    # Normally distributed learning rate, with mean .50 stddev 0.15, bounded between 0 and 1
    'eta': truncnorm(a=0, b=1, loc=0.50, scale=0.15),

    # Normally distributed subsample, with mean .50 stddev 0.15, bounded between 0 and 1
    'subsample': truncnorm(a=0, b=1, loc=0.50, scale=0.15)
}

# Instantiate the model
ml_model = XGBRegressor()

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
model_rf = random_search_rf.fit(cmips_x, cmips_pss)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('REGRESSION - PERCEIVED STRESS - RANDOM SEARCH - PCA - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_rf.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_rf.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_rf.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=277, eta=0.5276098469657554, max_depth=29, subsample=0.5677888025384356)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_pss,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

#######################################################################################################################

# region LINEAR REGRESSION - Potentially Traumatic Life Events

# Specify the hyperparameters of the linear regression
linear_reg = LinearRegression()

# Fit the linear regression with k-fold cross-validation
scores_linear_reg = cross_validate(estimator=linear_reg, X=cmips_x, y=cmips_lec,
                                   scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_explained_variance']), np.std(scores_linear_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_r2']), np.std(scores_linear_reg['test_r2'])))

# endregion

# region ELASTIC NET - Potentially Traumatic Life Events

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_lec,
                                    scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_explained_variance']), np.std(scores_elastic_net['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_r2']), np.std(scores_elastic_net['test_r2'])))

# endregion

# region LASSO REGRESSION - Potentially Traumatic Life Events

# Specify the hyperparameters of the lasso regression
lasso_reg = Lasso(alpha=0.5, max_iter=1000)

# Fit the lasso regression with k-fold cross-validation
scores_lasso_reg = cross_validate(estimator=lasso_reg, X=cmips_x, y=cmips_lec,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_explained_variance']), np.std(scores_lasso_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_r2']), np.std(scores_lasso_reg['test_r2'])))

# endregion

# region RIDGE REGRESSION - Potentially Traumatic Life Events

# Specify the hyperparameters of the ridge regression
ridge_reg = Ridge(alpha=0.5, max_iter=1000)

# Fit the ridge regression with k-fold cross-validation
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_lec,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

# region RANDOM FOREST - Potentially Traumatic Life Events

# Specify the hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=1.0)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_lec,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

# region XGBOOST - Potentially Traumatic Life Events

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=1000, eta=0.1, max_depth=10, subsample=1.0)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_lec,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - RIDGE REGRESSION - Potentially Traumatic Life Events

# Create the parameter search space
param_space = {
    # Randomly sample iterations
    'max_iter': randint(500, 1500),

    # Normally distributed alpha, with mean .50 stddev 0.15, bounded between 0 and 1
    'alpha': truncnorm(a=0, b=1, loc=0.50, scale=0.15)
}

# Instantiate the model
ml_model = Ridge()

# Create the random search algorithm
random_search_ridge = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='r2'
)

# Train the random search algorithm
model_ridge = random_search_ridge.fit(cmips_x, cmips_lec)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('REGRESSION - POTENTIALLY TRAUMATIC LIFE EVENTS - RANDOM SEARCH - PCA - RIDGE REGRESSION', file=f)
    print('\nBest Parameters', file=f)
    print(model_ridge.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_ridge.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_ridge.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify the hyperparameters of the ridge regression
ridge_reg = Ridge(alpha=0.6499344899146825, max_iter=1061)

# Fit the ridge regression with k-fold cross-validation
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_lec,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

#######################################################################################################################

# region LINEAR REGRESSION - Prejudiced Events

# Specify the hyperparameters of the linear regression
linear_reg = LinearRegression()

# Fit the linear regression with k-fold cross-validation
scores_linear_reg = cross_validate(estimator=linear_reg, X=cmips_x, y=cmips_dheq,
                                   scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_explained_variance']), np.std(scores_linear_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_r2']), np.std(scores_linear_reg['test_r2'])))

# endregion

# region ELASTIC NET - Prejudiced Events

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_dheq,
                                    scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_explained_variance']), np.std(scores_elastic_net['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_r2']), np.std(scores_elastic_net['test_r2'])))

# endregion

# region LASSO REGRESSION - Prejudiced Events

# Specify the hyperparameters of the lasso regression
lasso_reg = Lasso(alpha=0.5, max_iter=1000)

# Fit the lasso regression with k-fold cross-validation
scores_lasso_reg = cross_validate(estimator=lasso_reg, X=cmips_x, y=cmips_dheq,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_explained_variance']), np.std(scores_lasso_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_r2']), np.std(scores_lasso_reg['test_r2'])))

# endregion

# region RIDGE REGRESSION - Prejudiced Events

# Specify the hyperparameters of the ridge regression
ridge_reg = Ridge(alpha=0.5, max_iter=1000)

# Fit the ridge regression with k-fold cross-validation
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_dheq,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

# region RANDOM FOREST - Prejudiced Events

# Specify the hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=1.0)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_dheq,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

# region XGBOOST - Prejudiced Events

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=1000, eta=0.1, max_depth=10, subsample=1.0)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_dheq,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - RANDOM FOREST - Prejudiced Events

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
model_rf = random_search_rf.fit(cmips_x, cmips_dheq)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('REGRESSION - PREJUDICED EVENTS - RANDOM SEARCH - PCA - RANDOM FOREST', file=f)
    print('\nBest Parameters', file=f)
    print(model_rf.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_rf.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_rf.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=101, max_depth=22, max_features=0.5211175711876463)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_dheq,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

#######################################################################################################################

# region LINEAR REGRESSION - Identity Concealment

# Specify the hyperparameters of the linear regression
linear_reg = LinearRegression()

# Fit the linear regression with k-fold cross-validation
scores_linear_reg = cross_validate(estimator=linear_reg, X=cmips_x, y=cmips_oi,
                                   scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_explained_variance']), np.std(scores_linear_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_r2']), np.std(scores_linear_reg['test_r2'])))

# endregion

# region ELASTIC NET - Identity Concealment

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_oi,
                                    scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_explained_variance']), np.std(scores_elastic_net['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_r2']), np.std(scores_elastic_net['test_r2'])))

# endregion

# region LASSO REGRESSION - Identity Concealment

# Specify the hyperparameters of the lasso regression
lasso_reg = Lasso(alpha=0.5, max_iter=1000)

# Fit the lasso regression with k-fold cross-validation
scores_lasso_reg = cross_validate(estimator=lasso_reg, X=cmips_x, y=cmips_oi,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_explained_variance']), np.std(scores_lasso_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_r2']), np.std(scores_lasso_reg['test_r2'])))

# endregion

# region RIDGE REGRESSION - Identity Concealment

# Specify the hyperparameters of the ridge regression
ridge_reg = Ridge(alpha=0.5, max_iter=1000)

# Fit the ridge regression with k-fold cross-validation
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_oi,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

# region RANDOM FOREST - Identity Concealment

# Specify the hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=1.0)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_oi,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

# region XGBOOST - Identity Concealment

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=1000, eta=0.1, max_depth=10, subsample=1.0)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_oi,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - RANDOM FOREST - Identity Concealment

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
model_rf = random_search_rf.fit(cmips_x, cmips_oi)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('REGRESSION - IDENTITY CONCEALMENT - RANDOM SEARCH - PCA - RANDOM FOREST', file=f)
    print('\nBest Parameters', file=f)
    print(model_rf.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_rf.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_rf.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=527, max_depth=87, max_features=0.5470374899198611)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_oi,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

#######################################################################################################################

# region LINEAR REGRESSION - Internalized Stigma

# Specify the hyperparameters of the linear regression
linear_reg = LinearRegression()

# Fit the linear regression with k-fold cross-validation
scores_linear_reg = cross_validate(estimator=linear_reg, X=cmips_x, y=cmips_ihs,
                                   scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_explained_variance']), np.std(scores_linear_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_linear_reg['test_r2']), np.std(scores_linear_reg['test_r2'])))

# endregion

# region ELASTIC NET - Internalized Stigma

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_ihs,
                                    scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_explained_variance']), np.std(scores_elastic_net['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_r2']), np.std(scores_elastic_net['test_r2'])))

# endregion

# region LASSO REGRESSION - Internalized Stigma

# Specify the hyperparameters of the lasso regression
lasso_reg = Lasso(alpha=0.5, max_iter=1000)

# Fit the lasso regression with k-fold cross-validation
scores_lasso_reg = cross_validate(estimator=lasso_reg, X=cmips_x, y=cmips_ihs,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_explained_variance']), np.std(scores_lasso_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_lasso_reg['test_r2']), np.std(scores_lasso_reg['test_r2'])))

# endregion

# region RIDGE REGRESSION - Internalized Stigma

# Specify the hyperparameters of the ridge regression
ridge_reg = Ridge(alpha=0.5, max_iter=1000)

# Fit the ridge regression with k-fold cross-validation
scores_ridge_reg = cross_validate(estimator=ridge_reg, X=cmips_x, y=cmips_ihs,
                                  scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_explained_variance']), np.std(scores_ridge_reg['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_ridge_reg['test_r2']), np.std(scores_ridge_reg['test_r2'])))

# endregion

# region RANDOM FOREST - Internalized Stigma

# Specify the hyperparameters of the random forest
random_forest = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=1.0)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_ihs,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_explained_variance']), np.std(scores_random_forest['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_r2']), np.std(scores_random_forest['test_r2'])))

# endregion

# region XGBOOST - Internalized Stigma

# Specify the hyperparameters of the XGBoost
xg_boost = XGBRegressor(n_estimators=1000, eta=0.1, max_depth=10, subsample=1.0)

# Fit the elastic net with k-fold cross-validation
scores_xg_boost = cross_validate(estimator=xg_boost, X=cmips_x, y=cmips_ihs,
                                 scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_explained_variance']), np.std(scores_xg_boost['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_xg_boost['test_r2']), np.std(scores_xg_boost['test_r2'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - ELASTIC NET - Internalized Stigma

# Create the parameter search space
param_space = {
    # Randomly interations
    'max_iter': randint(500, 1500),

    # Normally distributed alpha, with mean .50 stddev 0.15, bounded between 0 and 1
    'alpha': truncnorm(a=0, b=1, loc=0.50, scale=0.15),

    # Normally distributed l1_ratio, with mean .50 stddev 0.15, bounded between 0 and 1
    'l1_ratio': truncnorm(a=0, b=1, loc=0.50, scale=0.15)
}

# Instantiate the model
ml_model = ElasticNet()

# Create the random search algorithm
random_search_en = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='r2'
)

# Train the random search algorithm
model_en = random_search_en.fit(cmips_x, cmips_ihs)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('REGRESSION - INTERNALIZED STIGMA - RANDOM SEARCH - PCA - ELASTIC NET', file=f)
    print('\nBest Parameters', file=f)
    print(model_en.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_en.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_en.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify the hyperparameters of the elastic net
elastic_net = ElasticNet(alpha=0.5384205455201654, l1_ratio=0.633333794769361, max_iter=1336)

# Fit the elastic net with k-fold cross-validation
scores_elastic_net = cross_validate(estimator=elastic_net, X=cmips_x, y=cmips_ihs,
                                    scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average explained variance: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_explained_variance']), np.std(scores_elastic_net['test_explained_variance'])))
print('Average R2: %.3f (%.3f)' % (
np.mean(scores_elastic_net['test_r2']), np.std(scores_elastic_net['test_r2'])))

# endregion

#######################################################################################################################
