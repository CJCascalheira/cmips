"""
# Author = Cory Cascalheira
# Date = 04/29/2024

The purpose of this script is to execute several ML models to compare performance AFTER performing featyre selection,
then performing hyperparameter tuning with random search on the best performing model.

This script classifies:
- Lifetime stressor count
- Lifetime stressor severity

RESOURCES
- https://ieeexplore.ieee.org/document/1453511
- https://pypi.org/project/pymrmr/
- https://home.penglab.com/papersall/docpdf/2005_TPAMI_FeaSel.pdf
- https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html

Original author: Hanchuan Peng (http://home.penglab.com/proj/mRMR/)

[1]: Hanchuan Peng, Fuhui Long, and Chris Ding, “Feature selection based on mutual information: criteria of
max-dependency, max-relevance, and min-redundancy,” IEEE Transactions on Pattern Analysis and Machine Intelligence,
Vol. 27, No. 8, pp.1226-1238, 2005.

The CPP code is subject to the original license (retrieved from http://home.penglab.com/proj/mRMR/FAQ_mrmr.htm):

The mRMR software packages can be downloaded and used, subject to the following conditions: Software and source code
Copyright (C) 2000-2007 Written by Hanchuan Peng. These software packages are copyright under the following conditions:
Permission to use, copy, and modify the software and their documentation is hereby granted to all academic and not-for-
profit institutions without fee, provided that the above copyright notice and this permission notice appear in all
copies of the software and related documentation and our publications (TPAMI05, JBCB05, CSB03, etc.) are appropriately
cited. Permission to distribute the software or modified or extended versions thereof on a not-for-profit basis is
explicitly granted, under the above conditions. However, the right to use this software by companies or other for profit
organizations, or in conjunction with for profit activities, and the right to distribute the software or modified or
extended versions thereof for profit are NOT granted except by prior arrangement and written consent of the copyright
holders. For these purposes, downloads of the source code constitute “use” and downloads of this source code by for
profit organizations and/or distribution to for profit institutions in explicitly prohibited without the prior consent
of the copyright holders. Use of this source code constitutes an agreement not to criticize, in any way, the code-
writing style of the author, including any statements regarding the extent of documentation and comments present. The
software is provided “AS-IS” and without warranty of any kind, expressed, implied or otherwise, including without
limitation, any warranty of merchantability or fitness for a particular purpose. In no event shall the authors be
liable for any special, incidental, indirect or consequential damages of any kind, or any damages whatsoever resulting
from loss of use, data or profits, whether or not advised of the possibility of damage, and on any theory of liability,
arising out of or in connection with the use or performance of these software packages.

The Python wrapper is subject to MIT license.
"""

# Load dependencies
import os
import random
import pandas as pd
import numpy as np
import math
import pymrmr
from sklearn.preprocessing import KBinsDiscretizer
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

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

# Initialize k-fold cross-validation
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Specify the metrics to use
my_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# endregion

#######################################################################################################################

# region

# Start with an ampty list
n_bins = []

# For loop to create bins
for i in range(len(cmips_x.columns)):

    # Get the bin edges using the Freedman Diaconis Estimator
    my_bind_edges = np.histogram_bin_edges(cmips_x.iloc[:, i].values, bins='fd')

    # If binwidth >= 1, then assign two
    if len(my_bind_edges) == 2:

        # Minus one to get number of bins
        my_n_bins = len(my_bind_edges)

        # Append number of bins to list
        n_bins.append(my_n_bins)

    elif len(my_bind_edges) > 2:

        # Minus one to get number of bins
        my_n_bins = len(my_bind_edges) - 1

        # Append number of bins to list
        n_bins.append(my_n_bins)

# Transform to features into matrices
cmips_x = cmips_x.values

# Discretize the data
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal').fit(cmips_x)
cmips_x_discrete = discretizer.transform(cmips_x)
cmips_x_discrete = cmips_x_discrete.astype(int)

# Prepare the discrete dataframe for mRMR
cmips_x_discrete = pd.DataFrame(cmips_x_discrete)

# Rename the columns
cmips_x_discrete.columns = cmips_x_df.columns

# Make copies of the dataframe
cmips_x_discrete_stressth = cmips_x_discrete.copy()
cmips_x_discrete_stressct = cmips_x_discrete.copy()

# Add labels to the first column of the dataframe
cmips_x_discrete_stressth.insert(0, 'cmips_stressth', cmips_stressth)
cmips_x_discrete_stressct.insert(0, 'cmips_stressct', cmips_stressct)

# Apply mRMR, keep 1/10 of the most informative features
n_feats_to_keep = math.ceil(len(cmips_x_discrete.columns) / 10)
mrmr_stressth = pymrmr.mRMR(cmips_x_discrete_stressth, 'MIQ', n_feats_to_keep)
mrmr_stressct = pymrmr.mRMR(cmips_x_discrete_stressct, 'MIQ', n_feats_to_keep)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_stressth]
cmips_x_2 = cmips_df[mrmr_stressct]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_stressth.csv")
cmips_x_2.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_stressct.csv")

# Transform features
cmips_x_1 = cmips_x_1.values
cmips_x_2 = cmips_x_2.values
cmips_stressth = cmips_stressth.values
cmips_stressct = cmips_stressct.values

# Standardize the feature matrix
sc = StandardScaler()
cmips_x_1 = sc.fit_transform(cmips_x_1)

sc = StandardScaler()
cmips_x_2 = sc.fit_transform(cmips_x_2)

# endregion

#######################################################################################################################

# region LOGISTIC REGRESSION - Lifetime Stress Severity

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=100)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=cmips_x_1, y=cmips_stressth,
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

# region SUPPORT VECTOR MACHINE - Lifetime Stress Severity

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=cmips_x_1, y=cmips_stressth,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# endregion

# region DECISION TREE - Lifetime Stress Severity

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x_1, y=cmips_stressth,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion

# region RANDOM FOREST - Lifetime Stress Severity

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x_1, y=cmips_stressth,
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

# region XGBOOST - Lifetime Stress Severity

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x_1, y=cmips_stressth,
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

# region LOGISTIC REGRESSION - Lifetime Stress Count

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=100)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=cmips_x_2, y=cmips_stressct,
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

# region SUPPORT VECTOR MACHINE - Lifetime Stress Count

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=cmips_x_2, y=cmips_stressct,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# endregion

# region DECISION TREE - Lifetime Stress Count

# Specify hyperparameters of the decision tree
dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)

# Fit the decision tree with k-fold cross-validation
scores_dt = cross_validate(estimator=dt, X=cmips_x_2, y=cmips_stressct,
                           scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('TRAINING METRICS')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_dt['test_accuracy']), np.std(scores_dt['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_dt['test_precision']), np.std(scores_dt['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_dt['test_recall']), np.std(scores_dt['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_dt['test_f1']), np.std(scores_dt['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_dt['test_roc_auc']), np.std(scores_dt['test_roc_auc'])))

# endregion

# region RANDOM FOREST - Lifetime Stress Count

# Specify hyperparameters of the random forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)

# Fit the random forest with k-fold cross-validation
scores_random_forest = cross_validate(estimator=random_forest, X=cmips_x_2, y=cmips_stressct,
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

# region XGBOOST - Lifetime Stress Count

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=cmips_x_2, y=cmips_stressct,
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
    print('TRAINING INFORMATION - RANDOM SEARCH - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify optimal hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.12596992965467246,
                    reg_lambda=8, max_depth=92, subsample=0.5952613310285284)

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
    print('TRAINING INFORMATION - RANDOM SEARCH - XGBOOST', file=f)
    print('\nBest Parameters', file=f)
    print(model_xgb.best_params_, file=f)
    print('\nBest Score', file=f)
    print(model_xgb.best_score_, file=f)
    print('\nBest Index', file=f)
    print(model_xgb.best_index_, file=f)
    print('\nAll Parameters', file=f)
    print('\n', file=f)

# Specify optimal hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.02137943371692049,
                    reg_lambda=3, max_depth=85, subsample=0.5809810172862997)

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
