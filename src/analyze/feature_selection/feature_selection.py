"""
# Author = Cory Cascalheira
# Date = 04/30/2024

The purpose of this script is to conduct feature selection to examine the top 100 most informative features.

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
import pymrmr
from sklearn.preprocessing import KBinsDiscretizer

# Set working directory
my_path = os.getcwd()

# Set the seed
random.seed(10)

#######################################################################################################################

# region mRMR Lifetime Stressor Count

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_strain.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_StressTH', 'label_StressCT'], axis=1)

# Get the label
cmips_y = cmips_df['label_StressCT']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_stressct.csv")

# endregion

# region mRMR Lifetime Stressor Severity

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_strain.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_StressTH', 'label_StressCT'], axis=1)

# Get the label
cmips_y = cmips_df['label_StressTH']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_stressth.csv")

# endregion

# region mRMR Perceived Stress

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_PSS_total', 'label_LEC_total', 'label_DHEQ_mean',
       'label_OI_mean', 'label_SOER_total', 'label_IHS_mean'], axis=1)

# Get the label
cmips_y = cmips_df['label_PSS_total']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_pss.csv")

# endregion

# region mRMR Potentially Traumatic Life Events

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_PSS_total', 'label_LEC_total', 'label_DHEQ_mean',
       'label_OI_mean', 'label_SOER_total', 'label_IHS_mean'], axis=1)

# Get the label
cmips_y = cmips_df['label_LEC_total']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_lec.csv")

# endregion

# region mRMR Prejudiced Events

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_PSS_total', 'label_LEC_total', 'label_DHEQ_mean',
       'label_OI_mean', 'label_SOER_total', 'label_IHS_mean'], axis=1)

# Get the label
cmips_y = cmips_df['label_DHEQ_mean']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_dheq.csv")

# endregion

# region mRMR Identity Concealment

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_PSS_total', 'label_LEC_total', 'label_DHEQ_mean',
       'label_OI_mean', 'label_SOER_total', 'label_IHS_mean'], axis=1)

# Get the label
cmips_y = cmips_df['label_OI_mean']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_oi.csv")

# endregion

# region mRMR Expected Rejection

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_PSS_total', 'label_LEC_total', 'label_DHEQ_mean',
       'label_OI_mean', 'label_SOER_total', 'label_IHS_mean'], axis=1)

# Get the label
cmips_y = cmips_df['label_SOER_total']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_soer.csv")

# endregion

# region mRMR Internalized Stigma

# Import the data
cmips_df = pd.read_csv(my_path + '/data/participants/for_analysis/for_models/cmips_class_qualtrics.csv')

# Get the features
cmips_x = cmips_df.drop(['participant_id', 'stress_posting', 'stress_n_content', 'high_freq_posting',
       'BSMAS_total', 'SMBS_total', 'CLCS_total', 'label_PSS_total', 'label_LEC_total', 'label_DHEQ_mean',
       'label_OI_mean', 'label_SOER_total', 'label_IHS_mean'], axis=1)

# Get the label
cmips_y = cmips_df['label_IHS_mean']

# Number of examples in each set
print(cmips_x.shape)
print(cmips_y.shape)

# Replace any NaN with zeros
cmips_x = pd.DataFrame(cmips_x).fillna(0)
cmips_x_df = cmips_x

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

# Add labels to the first column of the dataframe
cmips_x_discrete.insert(0, 'my_label', cmips_y)

# Apply mRMR, keep 100 of the most informative features
mrmr_results = pymrmr.mRMR(cmips_x_discrete, 'MIQ', 100)

# Select the reduced feature set
cmips_x_1 = cmips_df[mrmr_results]

# Export reduced features
cmips_x_1.to_csv(my_path + "/data/participants/for_analysis/for_models/reduced_features/feats_ihs.csv")

# endregion
