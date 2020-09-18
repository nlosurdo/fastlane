# import saspy
import sys
sys.path.insert(0,'C:\\PythonArea\\Lib\\site-packages')
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
rootout = 'C:\\PythonArea\\Data\\Output\\NearMiss1\\Model_Target'
subpath = '\\objects'


df = obj_from_pickle(dict_vars['pathinp'],
                     'L0_Input_Dataframe.pkl')

# Field conversion exception
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
fastlane.train_test_split(test_size=0.1)


# =============================================================================
# # <----- ----->
# # Feature Discovery
# # <----- ----->
# =============================================================================

fastlane.engineer_missing_imputer(SimpleImputer(strategy='mean'),
                                  ['fmtfloat'],
                                  mode='fmt_names',
                                  apply2test=False)

fastlane.df = fastlane.df.replace(np.nan, 'NA')

sampler = imblearn.under_sampling.RandomUnderSampler(
        random_state=0)
fastlane.df_balance(sampler)


subpath_edan = '\\Eda_Numericals'

fastlane.exploratory_df_numerical_plots(rootout, subpath_edan,
                                        smoothe_outlier=True)


subpath_edal = '\\Eda_Low_Categoricals'

fastlane.exploratory_df_lowcat_plots(rootout, subpath_edal)


subpath_edah = '\\Eda_High_Categoricals'

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
