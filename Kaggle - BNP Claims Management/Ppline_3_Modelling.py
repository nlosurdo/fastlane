# import saspy
import math
import pickle
import imblearn
import re
import pandas as pd
import numpy as np
import sklearn
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from tqdm import tqdm
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
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif,chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.ticker import FuncFormatter
dict_vars = variables()


# =============================================================================
# # <----- ----->
# # Importing Data Container
# # <----- ----->
# =============================================================================

rootin = dict_vars['pathinp']
rootout = dict_vars['pathout']
subpath = '\\reporting'
subpathII = '\\objects'

fastlane = obj_from_pickle(rootout + subpathII,
                           'dc_preprocessed.pkl')

df = fastlane.df
dftest = fastlane.dftest
yvals = fastlane.yvals
ytest = fastlane.ytest


# =============================================================================
# # <----- ----->
# # Model Training
# # <----- ----->
# =============================================================================

subpath_ms = subpath + '\\Mdl_ModelStatistics'

model_name = 'Gradient Boosting'

inner_classifier = GradientBoostingClassifier()

param_grid = {'max_depth': [3, 4, 5, 7, 11],
              'n_estimators': [100, 110, 120, 130]}

score = 'roc_auc'


classifier = GridSearchCV(inner_classifier,
                          param_grid=param_grid,
                          scoring=score, refit=True,
                          return_train_score=True,
                          cv=5, n_jobs=-1)

fastlane.model_fit_ingestion('CV Gradient Boosting',
                             classifier)


classifier = RandomForestClassifier()
fastlane.model_fit_ingestion('Random Forest',
                             classifier)


fastlane.models_plots(rootout, subpath_ms)


classifier = SVC(probability=True)
fastlane.model_fit_ingestion('Support Vector Machine',
                             classifier)


classifier = LogisticRegression()
fastlane.model_fit_ingestion('Logistic Regression',
                             classifier)


best_model = fastlane.model_set_best(mode='score_accuracy',
                                     path=subpathII,
                                     root=rootout,
                                     filename='best_model.pkl',
                                     returns=True)


# =============================================================================
# # <----- ----->
# # Explaiable AI
# # <----- ----->
# =============================================================================

subpath_eai = subpath + '\\Mdl_ExplainableAI'

fastlane.shap_fit_ingestion(
        apply='best_model',
        explainer='tree',
        tree_model_output='margin',
        tree_feature_perturbation='tree_path_dependent',
        keep_track=False,
        returns=False)

fastlane.shap_fit_ingestion(apply=['Random Forest'],
                            keep_track=True)


fastlane.shap_models_global_plots(rootout, subpath_eai)


fastlane.shap_models_dependence_plots(rootout, subpath_eai)


fastlane.shap_one_plots('CV Gradient Boosting',
                        rootout, subpath_eai,
                        key=('201912', 212505))
