"""
# Author = Cory Cascalheira
# Date = 04/30/2024

The purpose of this script is to execute the best ML model after PCA, then
performing hyperparameter tuning with random search on the best performing model.

This script classifies:
- Lifetime stressor count
- Lifetime stressor severity

RESOURCES
- https://xgboost.readthedocs.io/en/latest/parameter.html#
- https://jamesrledoux.com/code/randomized_parameter_search
- https://stackoverflow.com/questions/65666164/create-dataframe-with-multiple-arrays-by-column
- https://realpython.com/python-pretty-print/
- Scikit Learn documentation
- https://builtin.com/machine-learning/pca-in-python
"""

# Load dependencies
import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import truncnorm, randint, uniform
from sklearn.decomposition import PCA

# Set working directory
my_path = os.getcwd()

# Set the seed
random.seed(10)

# region PREPARE DATA AND METRICS

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_strain.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_StressTH', 'label_StressCT'], axis=1)

# Get the two labels
cmips_stressth = cmips_df['label_StressTH']
cmips_stressct = cmips_df['label_StressCT']

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
my_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# endregion

#######################################################################################################################

# region PCA

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

# region LOGISTIC REGRESSION - Lifetime Stressor Severity

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=500)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=cmips_x, y=cmips_stressth,
                                scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (
np.mean(scores_log_reg['test_accuracy']), np.std(scores_log_reg['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (
np.mean(scores_log_reg['test_precision']), np.std(scores_log_reg['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_log_reg['test_recall']), np.std(scores_log_reg['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_log_reg['test_f1']), np.std(scores_log_reg['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_log_reg['test_roc_auc']), np.std(scores_log_reg['test_roc_auc'])))

# endregion

# region SUPPORT VECTOR MACHINE - Lifetime Stressor Severity

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=cmips_x, y=cmips_stressth,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# endregion

# region DECISION TREE - Lifetime Stressor Severity

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x, y=cmips_stressth,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion

# region RANDOM FOREST - Lifetime Stressor Severity

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_stressth,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_accuracy']), np.std(scores_random_forest['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_precision']), np.std(scores_random_forest['test_precision'])))
print('Average recall: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_recall']), np.std(scores_random_forest['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_random_forest['test_f1']), np.std(scores_random_forest['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_roc_auc']), np.std(scores_random_forest['test_roc_auc'])))

# endregion

# region XGBOOST - Lifetime Stressor Severity

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_stressth,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

# endregion

#######################################################################################################################

# region LOGISTIC REGRESSION - Lifetime Stressor Count

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=500)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=cmips_x, y=cmips_stressct,
                                scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (
np.mean(scores_log_reg['test_accuracy']), np.std(scores_log_reg['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (
np.mean(scores_log_reg['test_precision']), np.std(scores_log_reg['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_log_reg['test_recall']), np.std(scores_log_reg['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_log_reg['test_f1']), np.std(scores_log_reg['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_log_reg['test_roc_auc']), np.std(scores_log_reg['test_roc_auc'])))

# endregion

# region SUPPORT VECTOR MACHINE - Lifetime Stressor Count

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=cmips_x, y=cmips_stressct,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# endregion

# region DECISION TREE - Lifetime Stressor Count

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x, y=cmips_stressct,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion

# region RANDOM FOREST - Lifetime Stressor Count

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_stressct,
                                      scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_accuracy']), np.std(scores_random_forest['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_precision']), np.std(scores_random_forest['test_precision'])))
print('Average recall: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_recall']), np.std(scores_random_forest['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_random_forest['test_f1']), np.std(scores_random_forest['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (
np.mean(scores_random_forest['test_roc_auc']), np.std(scores_random_forest['test_roc_auc'])))

# endregion

# region XGBOOST - Lifetime Stressor Count

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_stressct,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

# endregion

#######################################################################################################################

# region HYPERPARAMETER TUNING - RANDOM SEARCH - XGBOOST - Lifetime Stress Severity

# Create the parameter search space
param_space = {
    # Randomly sample L2 penalty
    'lambda': randint(1, 10),

    # Randomly sample numbers
    'max_depth': randint(10, 100),

    # Normally distributed subsample, with mean .50 stddev 0.15, bounded between 0 and 1
    'subsample': truncnorm(a=0, b=1, loc=0.50, scale=0.15),

    # Uniform distribution for learning rate
    'eta': uniform(0.001, 0.3)
}

# Instantiate the model
ml_model = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss')

# Create the random search algorithm
random_search_xgb = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='accuracy'
)

# Train the random search algorithm
model_xgb = random_search_xgb.fit(cmips_x, cmips_stressth)

# Save training results to file
with open(my_path + '/doc/random_search_output_classify_stressth.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('TRAINING INFORMATION - RANDOM SEARCH - PCA - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify optimal hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.2280036512989959,
                    reg_lambda=6, max_depth=21, subsample=0.624800304395675)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_stressth,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - XGBOOST - Lifetime Stress Count

# Create the parameter search space
param_space = {
    # Randomly sample L2 penalty
    'lambda': randint(1, 10),

    # Randomly sample numbers
    'max_depth': randint(10, 100),

    # Normally distributed subsample, with mean .50 stddev 0.15, bounded between 0 and 1
    'subsample': truncnorm(a=0, b=1, loc=0.50, scale=0.15),

    # Uniform distribution for learning rate
    'eta': uniform(0.001, 0.3)
}

# Instantiate the model
ml_model = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss')

# Create the random search algorithm
random_search_xgb = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='accuracy'
)

# Train the random search algorithm
model_xgb = random_search_xgb.fit(cmips_x, cmips_stressct)

# Save training results to file
with open(my_path + '/doc/random_search_output_classify_stressct.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('TRAINING INFORMATION - RANDOM SEARCH - PCA - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify optimal hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.073186745195608,
                    reg_lambda=3, max_depth=16, subsample=0.5872042319646955)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_stressct,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

# endregion
