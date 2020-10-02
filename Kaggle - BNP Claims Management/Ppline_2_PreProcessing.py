import pickle
import imblearn
import re
import pandas as pd
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from BorutaShap import BorutaShap

from Ppline_init import variables
from fastlane.data_ingestion import obj_to_pickle, obj_from_pickle
from fastlane.data_ingestion import pandas_profiling
from fastlane import BinaryLane
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif,chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
dict_vars = variables()

# =============================================================================
# # <----- ----->
# # Import DataFrame and Dict of columns, Pandas Profiling
# # <----- ----->
# =============================================================================
rootin = dict_vars['pathinp']
rootout = dict_vars['pathout']
subpath = '\\objects'

df = pd.read_csv(rootin+'\\train.csv')


# =============================================================================
# # <----- ----->
# # Data Preparation
# # <----- ----->
# =============================================================================
fastlane = BinaryLane(df, 'target', dict_vars)
fastlane.df.set_index('ID', inplace=True)

fastlane.set_categoricals(8)


fastlane.columns_keep(['key',
                       'fmtcategory',
                       'fmtfloat',
                       'fmtint'])

fastlane.columns_dropnan()


# =============================================================================
# # <----- ----->
# # Train - Test Split
# # <----- ----->
# =============================================================================

# Generate test set
fastlane.train_test_split(test_size=0.2)


# =============================================================================
# # <----- ----->
# # Data Encoding
# # <----- ----->
# =============================================================================

_fmtcategory = dict_vars['fmtcategory']

fastlane.df[_fmtcategory] = fastlane.df[_fmtcategory].replace(np.nan, 'NaN')

fastlane.encode_cathigh_set(ce.CatBoostEncoder(), apply2test=True)

fastlane.encode_catlow_set(ce.CatBoostEncoder(), apply2test=True)


# =============================================================================
# # <----- ----->
# # Feature Engeneering and Missing Imputation
# # <----- ----->
# =============================================================================

subpath_fp = '\\Feature Preparing'

fastlane.engineer_missing_set()

fastlane.engineer_smooth_outlier(['fmtfloat', 'fmtint'],
                                 multiplier=3,
                                 apply2test=True)

fastlane.engineer_standardize(apply2test=True)

fastlane.engineer_missing_imputer(IterativeImputer(),
                                  apply2test=True)

# fastlane.engineer_reset()


# =============================================================================
# # <----- ----->
# # Balancing Data
# # <----- ----->
# =============================================================================

subpath_fs = '\\Feature Selection'

fastlane.to_pickle(rootout, subpath, 'Iterative Imputed',
                   mode='standard')

# sampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
# sampler = imblearn.over_sampling.SMOTE(random_state=0)
sampler = imblearn.over_sampling.ADASYN(random_state=0)
# sampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
# sampler = imblearn.under_sampling.NearMiss(random_state=0)
# sampler = imblearn.under_sampling.ClusterCentroids(random_state=0)
# sampler = imblearn.under_sampling.EditedNearestNeighbours(random_state=0)
# sampler = imblearn.under_sampling.OneSidedSelection(random_state=0)
# sampler = imblearn.over_sampling.BorderlineSMOTE(random_state=0)
fastlane.df_balance(sampler)
# fastlane.set_prebalance()

fastlane.balance_plot(rootout, subpath_fp)

# =============================================================================
# # <----- ----->
# # Feature Selection
# # <----- ----->
# =============================================================================

selected_boruta = fastlane.selector_boruta(model=None,
                                           importance_measure='gini',
                                           classification=True,
                                           percentile=100,
                                           pvalue=0.05, n_trials=10,
                                           random_state=0,
                                           sample=True, returns=True)

fastlane.selector_feature_plot(rootout, subpath_fs, selector='boruta',
                               plot_threshold=0)

fastlane.to_pickle(rootout, subpath, 'Boruta selector',
                   mode='standard')

fastlane.selector_apply(selector='boruta')

# classifier = RandomForestClassifier(n_estimators=100, criterion='gini',
#                                     max_depth=None, max_features='auto',
#                                     min_impurity_decrease=0.0,
#                                     bootstrap=True, n_jobs=-1,
#                                     random_state=0)

# selected_model = fastlane.selector_kfold_model(classifier,
#                                                threshold=0,
#                                                returns=True)

# fastlane.selector_feature_plot(rootout, subpath_fs, selector='model',
#                                plot_threshold=-0.5)


# =============================================================================
# # <----- ----->
# # Explaiable Feature Selection
# # <----- ----->
# =============================================================================

classifier = RandomForestClassifier(n_estimators=100, criterion='gini',
                                    max_depth=None, max_features='auto',
                                    min_impurity_decrease=0.0,
                                    bootstrap=True, n_jobs=-1,
                                    random_state=0)

explained_rfs = fastlane.recursive_selector(
        classifier, mode='cv',
        start_slot=5, threshold=-1,
        cv=5,
        n_splits=5,
        shuffle=True,
        random_state=0,
        returns=True)

fastlane.recursive_selector_plot(rootout, subpath_fs, 'explained_frs',
                                 filtercols=1)


fastlane.columns_drop_correlated()


