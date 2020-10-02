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
# # Import DataFrame and Dict of columns, Pandas Profiling
# # <----- ----->
# =============================================================================
rootin = dict_vars['pathinp']
rootout = dict_vars['pathout']
subpath = '\\objects'

fastlane = obj_from_pickle(rootout + subpath, 'Boruta selector')

df = fastlane.df
dftest = fastlane.dftest
yvals = fastlane.yvals
ytest = fastlane.ytest


# =============================================================================
# # <----- ----->
# # Model Training
# # <----- ----->
# =============================================================================

subpath_ms = '\\Machine Learning Models'

model_name = 'Gradient Boosting'

inner_classifier = GradientBoostingClassifier()

param_grid = {'max_depth': [3, 5, 7],
              'n_estimators': [100, 130]}

score = 'neg_log_loss'

classifier = GridSearchCV(inner_classifier,
                          param_grid=param_grid,
                          scoring=score, refit=True,
                          return_train_score=True,
                          cv=2, n_jobs=-1)

fastlane.model_fit_ingestion('CV Gradient Boosting',
                             classifier)

fastlane.models_plots(rootout, subpath_ms)

classifier = RandomForestClassifier()
fastlane.model_fit_ingestion('Random Forest',
                             classifier)

classifier = LogisticRegression()
fastlane.model_fit_ingestion('Logistic Regression',
                             classifier)

classifier = GradientBoostingClassifier()
fastlane.model_fit_ingestion('Gradient Boosting Classifier',
                             classifier)

fastlane.models_plots(rootout, subpath_ms)

# classifier = SVC(probability=True)
# fastlane.model_fit_ingestion('Support Vector Machine',
#                              classifier)

best_model = fastlane.model_set_best(mode='score_neg_log_loss',
                                     path=subpath,
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
