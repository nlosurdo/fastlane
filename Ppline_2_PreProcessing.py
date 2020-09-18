import sys
sys.path.insert(0, 'C:\\PythonArea\\Lib\\site-packages')
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

from pipiline__init__ import variables
from fastlane.data_ingestion import obj_to_pickle, obj_from_pickle
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
# # Import DataFrame and Dict of columns
# # <----- ----->
# =============================================================================

rootin = dict_vars['pathinp']
rootout = dict_vars['pathout']
subpath = '\\reporting'
subpathII = '\\objects'

df = obj_from_pickle(dict_vars['pathinp'],
                     'L0_Input_Dataframe.pkl')


dict_cols = obj_from_pickle(dict_vars['pathinp'],
                            'L0_DictofColumns.pkl')

dict_cols['key'] = ['PERIODO', 'ID_POL']


# =============================================================================
# # <----- ----->
# # Data Preparation
# # <----- ----->
# =============================================================================

fastlane = BinaryLane(df, 'RETENTION', dict_cols)


fastlane.set_categoricals(8)


fastlane.columns_keep(['fmtcategory',
                      'fmtordcategory',
                       'fmtfloat',
                       'fmtint'])


fastlane.columns_dropnan()
fastlane.columns_drop(['n_JUMPRC'])


# =============================================================================
# #<----- ----->
# # Feature Creation
# #<----- ----->
# =============================================================================

df = fastlane.get()

df['dlt_PrRCA_Prmkt'] = df['PREMIO_PROPOSTO_RCA'] -\
    df['Market_Premium']

df['SUBAGENZIA'] = df['AGENZIA'] +\
    '_' + df['SUBAGENZIA']

df['Q4_FK_COD_MODELLO'] = df['Q4_FK_COD_MARCA'] +\
    '_' + df['Q4_FK_COD_MODELLO']

fastlane.load(df,
              inp_dict_cols={'fmtfloat': ['dlt_PrRCA_Prmkt']})


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

fastlane.encode_cathigh_set(ce.TargetEncoder(), enc_type='kfold',
                            n_splits=5,
                            apply2test=True)

fastlane.encode_catlow_set(ce.OneHotEncoder(),
                           apply2test=True)

fastlane.encode_catord_set(ce.OrdinalEncoder(),
                           apply2test=True)

# =============================================================================
# # <----- ----->
# # Feature Engeneering and Missing Imputation
# # <----- ----->
# =============================================================================

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

subpath_fs = subpath + '\\Fs_FeatureSelection'


# sampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
# sampler = imblearn.over_sampling.SMOTE(random_state=0)
# sampler = imblearn.over_sampling.ADASYN(random_state=0)
sampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
# sampler = imblearn.under_sampling.NearMiss(random_state=0)
# sampler = imblearn.under_sampling.ClusterCentroids(random_state=0)
# sampler = imblearn.under_sampling.EditedNearestNeighbours(random_state=0)
# sampler = imblearn.under_sampling.OneSidedSelection(random_state=0)
# sampler = imblearn.over_sampling.BorderlineSMOTE(random_state=0)
fastlane.df_balance(sampler)
# fastlane.set_prebalance()

fastlane.balance_plot(rootout, subpath_fs)

# =============================================================================
# # <----- ----->
# # Feature Selection
# # <----- ----->
# =============================================================================


classifier = RandomForestClassifier(n_estimators=100, criterion='gini',
                                    max_depth=None, max_features='auto',
                                    min_impurity_decrease=0.0,
                                    bootstrap=True, n_jobs=-1,
                                    random_state=0)

selected_model = fastlane.selector_kfold_model(classifier,
                                               threshold=0,
                                               returns=True)

fastlane.selector_feature_plot(rootout, subpath_fs, selector='model',
                               plot_threshold=-0.5)


selected_boruta = fastlane.boruta_selector(model=None,
                                           importance_measure='gini',
                                           classification=True,
                                           percentile=100,
                                           pvalue=0.05, n_trials=30,
                                           random_state=0,
                                           sample=True, returns=True)

fastlane.selector_feature_plot(rootout, subpath_fs, selector='model',
                               plot_threshold=0)


fastlane.selector_apply(selector='model')


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

# =============================================================================
# # <----- ----->
# # Export Data Container
# # <----- ----->
# =============================================================================

obj_to_pickle(fastlane, rootout + subpathII,
              'dc_preprocessed.pkl')
