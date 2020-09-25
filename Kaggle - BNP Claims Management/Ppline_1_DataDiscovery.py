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
subpath_edan = '\\Data Discovery Pandas Profiling'

df = pd.read_csv(rootin+'\\train.csv')

# chunks = 20

# for i in range(0, len(df.columns), chunks):

#     dfi = df.iloc[:, i:i+chunks]

#     pandas_profiling(dfi, rootout+subpath_edan+'\\Pandas-Profiling_'
#                      + str(i) + '_' + str(i+chunks) + '_.html')


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
# #<----- ----->
# # Feature Creation
# #<----- ----->
# =============================================================================

# df = fastlane.get()
# fastlane.load(df, inp_dict_cols={'...': ['...']})


# =============================================================================
# # <----- ----->
# # Train - Test Split
# # <----- ----->
# =============================================================================

# Generate test set
fastlane.train_test_split(test_size=0.2)


# =============================================================================
# # <----- ----->
# # Preprocessing
# # <----- ----->
# =============================================================================

fastlane.engineer_missing_imputer(SimpleImputer(strategy='mean'),
                                  ['fmtfloat'],
                                  mode='fmt_names',
                                  apply2test=False)

fastlane.engineer_missing_imputer(SimpleImputer(strategy='constant',
                                                fill_value=-1),
                                  ['fmtint'],
                                  mode='fmt_names',
                                  apply2test=False)

_fmtcategory = dict_vars['fmtcategory']

fastlane.df[_fmtcategory] = fastlane.df[_fmtcategory].replace(np.nan, 'NaN')

sampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
fastlane.df_balance(sampler)


# =============================================================================
# # <----- ----->
# # Data Discovery
# # <----- ----->
# =============================================================================

subpath_edan = '\\Data Discovery Numericals'

fastlane.exploratory_df_numerical_plots(rootout, subpath_edan,
                                        smoothe_outlier=True)


subpath_edal = '\\Data Discovery Low Categoricals'

fastlane.exploratory_df_lowcat_plots(rootout, subpath_edal)


subpath_edah = '\\Data Discovery High Categoricals'

fastlane.exploratory_df_highcat_plots(rootout, subpath_edah)


fastlane.set_prebalance()


# =============================================================================
# # <----- ----->
# # Feature Selection
# # <----- ----->
# =============================================================================

# Univariate Categorical Feature Selection
selected_cat = fastlane.selector_univariate_cat(
        returns=True)

# Univariate Numerical Feature Selection
selected_num = fastlane.selector_univariate_num(
        returns=True)
