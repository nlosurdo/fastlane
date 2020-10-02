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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
dict_vars = variables()

# =============================================================================
# # <----- ----->
# # Import DataFrames
# # <----- ----->
# =============================================================================
rootin = dict_vars['pathinp']
rootout = dict_vars['pathout']
subpath = '\\objects'

fastlane = obj_from_pickle(rootout + subpath, 'Iterative Imputed')
dfprod = pd.read_csv(rootin+'\\test.csv')


# =============================================================================
# # <----- ----->
# # Production Exec
# # <----- ----->
# =============================================================================
fastlane.production_df_load(dfprod)


_fmtcategory = dict_vars['fmtcategory']

fastlane.dfprod[_fmtcategory] = fastlane.dfprod[
    _fmtcategory].replace(np.nan, 'NaN')


fastlane.production_exec(['encode_cathigh_set', 'encode_catlow_set',
                          'engineer_missing_set', 'engineer_smooth_outlier',
                          'engineer_standardize', 'engineer_missing_imputer'])

