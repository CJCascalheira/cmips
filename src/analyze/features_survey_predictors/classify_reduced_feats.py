"""
# Author = Cory Cascalheira
# Date = 05/02/2024

The purpose of this script is to add survey-based predictors to models using the reduced feature space.
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

# region SET PATH AND PREPARE METRICS

# Set working directory
my_path = os.getcwd()

# Set the seed
random.seed(10)

# Initialize k-fold cross-validation
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Specify the metrics to use
my_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# endregion

#######################################################################################################################

#######################################################################################################################
# ADULT STRAIN
#######################################################################################################################

#######################################################################################################################

# region DATA FOR ADULT STRAIN

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_strain.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'label_StressTH', 'label_StressCT', 'stress_posting', 'stress_n_content',
                         'high_freq_posting', 'BSMAS_total', 'SMBS_total', 'CLCS_total'], axis=1)
cmips_x_new = cmips_df[['stress_posting', 'stress_n_content', 'high_freq_posting', 'BSMAS_total', 'SMBS_total', 'CLCS_total']]

# Get the two labels
cmips_stressth = cmips_df['label_StressTH']
cmips_stressct = cmips_df['label_StressCT']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_stressth.shape)
print(cmips_stressct.shape)

# Transform to matrices
cmips_x = cmips_x.values
cmips_x_new = cmips_x_new.values
cmips_stressth = cmips_stressth.values
cmips_stressct = cmips_stressct.values

# Standardize the feature matrix
sc = StandardScaler()
cmips_x = sc.fit_transform(cmips_x)

sc = StandardScaler()
cmips_x_new = sc.fit_transform(cmips_x_new)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x = cmips_x.values

cmips_x_new = pd.DataFrame(cmips_x_new).fillna(0)
cmips_x_new = cmips_x_new.values

# endregion

# region CONDUCT PCA

# Initialize PCA and keep 95% of the variance
pca = PCA(n_components=0.95)

# Fit the PCA model
pca.fit(cmips_x)

# How many components?
print(pca.n_components_)

# Transform the feature space
cmips_x = pca.transform(cmips_x)

# Combine with new values
cmips_x = pd.concat([pd.DataFrame(cmips_x), pd.DataFrame(cmips_x_new)], axis=1)
cmips_x = cmips_x.values

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
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('CLASSIFICATION - LIFETIME STRESSOR COUNT - RANDOM SEARCH - PCA - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify optimal hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.16599099197817904,
                    reg_lambda=5, max_depth=92, subsample=0.5623187233605648)

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
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('CLASSIFICATION - LIFETIME STRESSOR SEVERITY - RANDOM SEARCH - PCA - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify optimal hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.2589688535596577,
                    reg_lambda=5, max_depth=47, subsample=0.593860473250242)

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

#######################################################################################################################
# QUALTRICS SURVEY
#######################################################################################################################

#######################################################################################################################

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'label_PSS_total', 'label_LEC_total', 'label_DHEQ_mean','label_OI_mean',
                         'label_SOER_total', 'label_IHS_mean', 'stress_posting', 'stress_n_content', 'high_freq_posting',
                         'BSMAS_total', 'SMBS_total', 'CLCS_total'], axis=1)
cmips_x_new = cmips_df[['stress_posting', 'stress_n_content', 'high_freq_posting', 'BSMAS_total', 'SMBS_total', 'CLCS_total']]

# Get the two labels
cmips_soer = cmips_df['label_SOER_total']
cmips_dheq = cmips_df['label_DHEQ_mean']
cmips_ihs = cmips_df['label_IHS_mean']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_soer.shape)
print(cmips_dheq.shape)
print(cmips_ihs.shape)

# Transform to matrices
cmips_x = cmips_x.values
cmips_x_new = cmips_x_new.values
cmips_soer = cmips_soer.values
cmips_dheq = cmips_dheq.values
cmips_ihs = cmips_ihs.values

# Standardize the feature matrix
sc = StandardScaler()
cmips_x = sc.fit_transform(cmips_x)

sc = StandardScaler()
cmips_x_new = sc.fit_transform(cmips_x_new)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x = cmips_x.values

cmips_x_new = pd.DataFrame(cmips_x_new).fillna(0)
cmips_x_new = cmips_x_new.values

# endregion

# region CONDUCT PCA

# Initialize PCA and keep 95% of the variance
pca = PCA(n_components=0.95)

# Fit the PCA model
pca.fit(cmips_x)

# How many components?
print(pca.n_components_)

# Transform the feature space
cmips_x = pca.transform(cmips_x)

# Combine with new values
cmips_x = pd.concat([pd.DataFrame(cmips_x), pd.DataFrame(cmips_x_new)], axis=1)
cmips_x = cmips_x.values

# endregion

#######################################################################################################################

# region LOGISTIC REGRESSION - Prejudiced Events

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=500)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=cmips_x, y=cmips_dheq,
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

# region SUPPORT VECTOR MACHINE - Prejudiced Events

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=cmips_x, y=cmips_dheq,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# endregion

# region DECISION TREE - Prejudiced Events

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x, y=cmips_dheq,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion

# region RANDOM FOREST - Prejudiced Events

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_dheq,
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

# region XGBOOST - Prejudiced Events

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_dheq,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - XGBOOST - Prejudiced Events

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
model_xgb = random_search_xgb.fit(cmips_x, cmips_dheq)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('CLASSIFICATION - PREJUDICED EVENTS - RANDOM SEARCH - PCA - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify optimal hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.09894385433227045,
                    reg_lambda=6, max_depth=10, subsample=0.5097871788583676)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_dheq,
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

# region LOGISTIC REGRESSION - Expected Rejection

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=500)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=cmips_x, y=cmips_soer,
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

# region SUPPORT VECTOR MACHINE - Expected Rejection

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=cmips_x, y=cmips_soer,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# endregion

# region DECISION TREE - Expected Rejection

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x, y=cmips_soer,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion

# region RANDOM FOREST - Expected Rejection

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_soer,
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

# region XGBOOST - Expected Rejection

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_soer,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - RANDOM FOREST - EXPECTED REJECTION

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
ml_model = RandomForestClassifier()

# Create the random search algorithm
random_search_rf = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='accuracy'
)

# Train the random search algorithm
model_rf = random_search_rf.fit(cmips_x, cmips_soer)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('CLASSIFICATION - EXPECTED REJECTION - RANDOM SEARCH - PCA - RANDOM FOREST', file=f)
    print('\nBest Parameters', file=f)
    print(model_rf.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_rf.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_rf.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=292, criterion='entropy', max_depth=64,
                                       max_features=0.5630181739176333)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_soer,
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

#######################################################################################################################

# region LOGISTIC REGRESSION - Internalized Stigma

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=500)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=cmips_x, y=cmips_ihs,
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

# region SUPPORT VECTOR MACHINE - Internalized Stigma

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=cmips_x, y=cmips_ihs,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# endregion

# region DECISION TREE - Internalized Stigma

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x, y=cmips_ihs,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion

# region RANDOM FOREST - Internalized Stigma

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x, y=cmips_ihs,
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

# region XGBOOST - Internalized Stigma

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x, y=cmips_ihs,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

# endregion

# region HYPERPARAMETER TUNING - RANDOM SEARCH - DECISION TREE - Internalized Stigma

# Create the parameter search space
param_space = {
    # Randomly sample numbers
    'max_depth': randint(10, 100),

    # Impurity functions to use
    'criterion': ['gini', 'entropy'],
}

# Instantiate the model
ml_model = DecisionTreeClassifier(random_state=1)

# Create the random search algorithm
random_search_dt = RandomizedSearchCV(
    estimator=ml_model,
    param_distributions=param_space,
    n_iter=100,
    scoring=my_metrics,
    cv=kfold,
    refit='accuracy'
)

# Train the random search algorithm
model_dt = random_search_dt.fit(cmips_x, cmips_ihs)

# Save training results to file
with open(my_path + '/doc/random_search_output_final.txt', 'a') as f:
    print('\n#############################################################', file=f)
    print('CLASSIFICATION - INTERNALIZED STIGMA - RANDOM SEARCH - PCA - DECISION TREE', file=f)
    print('\nBest Parameters', file=f)
    print(model_dt.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_dt.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_dt.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=45, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x, y=cmips_ihs,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion
