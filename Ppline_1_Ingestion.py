import sys
sys.path.insert(0,'C:\\PythonArea\\Lib\\site-packages')
sys.path.insert(1,'C:\\PythonArea\\Scripts\\[1] Aviva Retention Auto')
import saspy
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
from fastlane.data_ingestion import open_sas, import_sas_contents
from fastlane.data_ingestion import ddfs2ddfs_pdtypes, ddfs2dict_warnings
from fastlane.data_ingestion import ddfs_engeneering, ddfs_apply_pdtype
from fastlane.data_ingestion import ddfs2df_append, pandas_profiling
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


# <----- ----->
# Data Ingestion
# <----- ----->


# Import Sas Tables in a Dictionary with Df and Col Lists (from SAS labels)
with open_sas() as sas:

    dict_ds2df = import_sas_contents(sas, dict_vars['sastables'],
                                     dict_vars['saslibrefs'],
                                     dict_vars['sasoptions'])


# Add Grouped Mapped Pandas Dtype Columns (3 args)
dict_ds2df = ddfs2ddfs_pdtypes(dict_ds2df, dict2fmt_update={}, Update=False)

# Controls df (warning dict)
dict_wrn = ddfs2dict_warnings(dict_ds2df)

# Transform DataFrames
dict_dfnew = ddfs_engeneering(dict_ds2df, dict_vars['df_mapfunc'])
dict_wrn_new = ddfs2dict_warnings(dict_dfnew)

# Apply Formats to Pandas df Columns
dict_dfnew = ddfs_apply_pdtype(dict_dfnew)

# Creating one Table (Merging all input dataframes)
df, dict_cols = ddfs2df_append(dict_dfnew,
                               dict_vars['sequential_joins'],
                               df_2pickle=dict_vars['pathinp'],
                               dictcols_return=True)

# Pandas Profiling Report
#pandas_profiling(df, dict_vars['pathout'] +
#                 dict_vars['pathimg'] +
#                 '\\PandasProfiling_Report.html')

del dict_ds2df, dict_wrn, dict_dfnew, dict_wrn_new
