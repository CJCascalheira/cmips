"""
Author = Cory J. Cascalheira
Date = 04/23/2024

The purpose of this script is twofold: (1) to clean up the emojis in the stress & relaxation dataset and (2) to train
generic ML classifiers to predict stress and relaxation.

# References
- https://medium.com/@sarahisdevs/convert-emoji-into-text-in-python-c2afdfd94ab4
- https://towardsdatascience.com/emojis-aid-social-media-sentiment-analysis-stop-cleaning-them-out-bb32a1e5fc8e
- https://www.sciencedirect.com/science/article/abs/pii/S0306457316302321
"""

#region LOAD AND IMPORT

# General dependencies
import os
import pandas as pd
import numpy as np
import emoji
import random

# Import machine learning packages
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Set file path
my_path = os.getcwd()

# Set the seed
random.seed(10)

# Import data
stress_relax_df = pd.read_csv(my_path + '/data/util/tensi_strength_data/stress_relax_df.csv')
stress_relax_df_ngrams = pd.read_csv(my_path + "/data/util/tensi_strength_data/stress_relax_df_ngrams.csv")
cmips_df_ngrams = pd.read_csv(my_path + "/data/util/tensi_strength_data/cmips_df_ngrams.csv")

#endregion

#region COVERT EMOJIS TO TEXTS

# Transform emojis in all columns
stress_relax_df['text'] = stress_relax_df['text'].astype(str)
stress_relax_df['text'] = stress_relax_df['text'].map(lambda x: emoji.demojize(x))

# Save file
stress_relax_df.to_csv(my_path + '/data/util/tensi_strength_data/stress_relax_df_demojized.csv')

#endregion

#region MACHINE LEARNING PREPROCESSING

# Select the features and labels
srdf_relax_y = stress_relax_df_ngrams['relax']
srdf_stress_y = stress_relax_df_ngrams['stress']
srdf_features = stress_relax_df_ngrams.drop(['relax', 'stress', 'post_id', 'text'], axis=1)

# Check the shape of the data
print(srdf_relax_y.shape)
print(srdf_stress_y.shape)
print(srdf_features.shape)

# Transform to matrices
srdf_relax_y = srdf_relax_y.values
srdf_stress_y = srdf_stress_y.values
srdf_features = srdf_features.values

# Preprocess CMIPS features
cmips_features = cmips_df_ngrams.drop(['participant_id', 'timestamp', 'text'], axis=1)
cmips_features = cmips_features.values

# Initialize k-fold cross-validation
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Specify the metrics to use
my_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

#endregion

#region MACHINE LEARNING CLASSIFIER TRAINING - RELAX

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=100)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=srdf_features, y=srdf_relax_y,
                                scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('METRICS - RELAX DATA')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_log_reg['test_accuracy']), np.std(scores_log_reg['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_log_reg['test_precision']), np.std(scores_log_reg['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_log_reg['test_recall']), np.std(scores_log_reg['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_log_reg['test_f1']), np.std(scores_log_reg['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_log_reg['test_roc_auc']), np.std(scores_log_reg['test_roc_auc'])))

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=srdf_features, y=srdf_relax_y,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('METRICS - RELAX DATA')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=srdf_features, y=srdf_relax_y,
               scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('METRICS - RELAX DATA')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

#endregion

#region MACHINE LEARNING CLASSIFIER TRAINING - STRESS

# Specify the hyperparameters of the logistic regression
log_reg = LogisticRegression(penalty='l2', C=1.0, max_iter=100)

# Fit the logistic regression with k-fold cross-validation
scores_log_reg = cross_validate(estimator=log_reg, X=srdf_features, y=srdf_stress_y,
                                scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('METRICS - STRESS DATA')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_log_reg['test_accuracy']), np.std(scores_log_reg['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_log_reg['test_precision']), np.std(scores_log_reg['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_log_reg['test_recall']), np.std(scores_log_reg['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_log_reg['test_f1']), np.std(scores_log_reg['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_log_reg['test_roc_auc']), np.std(scores_log_reg['test_roc_auc'])))

# Specify hyperparameters of the SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)

# Fit the SVM with k-fold cross-validation
scores_svm = cross_validate(estimator=svm, X=srdf_features, y=srdf_stress_y,
                            scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('METRICS - STRESS DATA')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_svm['test_accuracy']), np.std(scores_svm['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_svm['test_precision']), np.std(scores_svm['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_svm['test_recall']), np.std(scores_svm['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_svm['test_f1']), np.std(scores_svm['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_svm['test_roc_auc']), np.std(scores_svm['test_roc_auc'])))

# Specify hyperparameters of the XGBoost
xgb = XGBClassifier(booster='gbtree', use_label_encoder=False, eval_metric='logloss', eta=0.1, max_depth=10)

# Fit the XGBoost classifier with k-fold cross-validation
scores_xgb = cross_validate(estimator=xgb, X=srdf_features, y=srdf_stress_y,
               scoring=my_metrics, cv=kfold, n_jobs=-1, error_score='raise')

# Print the average scores during training
print('METRICS - STRESS DATA')
print('Average accuracy: %.3f (%.3f)' % (np.mean(scores_xgb['test_accuracy']), np.std(scores_xgb['test_accuracy'])))
print('Average precision: %.3f (%.3f)' % (np.mean(scores_xgb['test_precision']), np.std(scores_xgb['test_precision'])))
print('Average recall: %.3f (%.3f)' % (np.mean(scores_xgb['test_recall']), np.std(scores_xgb['test_recall'])))
print('Average F1: %.3f (%.3f)' % (np.mean(scores_xgb['test_f1']), np.std(scores_xgb['test_f1'])))
print('Average ROC AUC: %.3f (%.3f)' % (np.mean(scores_xgb['test_roc_auc']), np.std(scores_xgb['test_roc_auc'])))

#endregion

#region MACHINE LABELING

# Fit the LR classifier to the data
log_reg.fit(srdf_features, srdf_stress_y)

# Make predictions
primary_preds = log_reg.predict(cmips_features)
print(len(primary_preds))

# Append to dataframe
cmips_df = cmips_df_ngrams[['participant_id', 'timestamp']]
cmips_df['feat_stress_classifier'] = primary_preds

# Save file
cmips_df.to_csv(my_path + "/data/participants/features/cmips_feature_set_03.csv")

#endregion
