import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
import shap
import os
import pickle
import pandas_profiling as pp
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold
from .data_ingestion import obj_to_pickle, obj_from_pickle
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif,chi2, mutual_info_classif
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
from BorutaShap import BorutaShap


# Decorator to ingest external functions into the fastlane class
def ingest_function(inp_func):
    """
    This function 'decorator' enable the user to ingest custom operations
    in the  fastlane pipiline, so that thay can be scheduled along with the
    other standard fastlane methods.
    1) In order to make it works the custom function must be referenced by
    this ingest_function decorator - @ingest_function
    2) The second mandatory rules to follow, are those one of using the
    fastlane get() method at th very begining of the user function, in order
    to get out the dataframe from the fastlane object and the load() method
    in order to reload it in the standard pipiline at the end of the function.
    Once these two operations are made, the user function can be inserted
    in the compilied pipiline in the following straightforward way:
        - function_name(fastlane_object_name)
    """
    def wrapper(obj, setslot=-1):

        dict_locals = {}

        obj._set_schedule_flow(inp_func,
                               dict_locals,
                               setslot)

    return wrapper


class BinaryLane():

    def __init__(self, df, y, inp_dict_cols,
                 setslot=None):
        """
         Class Constructor

         Class to speed up machine learning pipiline developings, for binary
         supervised classification projects.
         The class embedds many sklearn, pandas, numpy functions and methods
         in an importance selected and semplified way, which will guides the
         developer during making the specific ml project pipiline.

         Builds the object by setting the required attributes, which are:
             - A Pandas DataFrame that will be the core data reference for
             each of the methods implemented in the class.
             - Pandas data Type groups with list of df columns in them.
             Pandas dtypes accepted:
             Note that at least one group is required to be given, to make the
             class works, along with the key group, whose columns are required
             to be set as index data frame) and 'lstYtarget', containing the y
             target related fields:
                 'key' : it is the only non pdtype attr., the key of the table
                 'fmtcategory' : categorical columns
                 'fmtordcategory' : ordinal categorical columns
                 'fmtint' : integer number columns
                 'fmtfloat' : float number columns
                 'fmtdatetime' : date and datetime columns

         Parameters
         _ _ _ _ _
         df : Pandas DataFrame
             The input table
         y : Strng
             Name of the binary targer columns
         inp_dict_cols : dictionary
             The input Dictionary containing keys: 'Pandas Data Type Group',
             values: 'list of df column names'
        setslot : integer
             By assigning a value to the parameter, the scheduling compilation
             mode is activated. The method is so inserted in the object flow
             array with the assigned slot position, therefore compilied and
             not executed at run time. Please note that, the given integer
             position must be coherent with the previous assigned ones,
             indeed it must respect the ordinal sequence execution.
             By definition, this method must always be the first one to be
             inserted in the flow. It is also possibile to use the parameter
             by using a dafault value, so that the flow scheduling order is
             inferred by the execution order of the object methods at
             the run time. The default values is -1.
             It is also possibile to overwrite an already assigned flow
             position, by giving the parameter the same slot as that one of
             the method we want to replace.

         Returns
         _ _ _ _ _
         Object istance attributes of the class:
            self.df : dataframe
            self.y : y target name
            self.yvals : y target series
            self.dict_cols : dictionary of dtype colums
            6 self.fmt_pdytpes : obj. attribute pdtype lists
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.__init__,
                                    dict_locals,
                                    setslot)
            return

        self._df_init_ = df.drop(columns=y).sort_index().copy()
        self.df = df.drop(columns=y).sort_index().copy()
        self.y = y
        self.yvals = df[y].sort_index().copy()
        inp_dict_cols = inp_dict_cols.copy()

        pandas_dtypes = ['key', 'fmtcategory', 'fmtordcategory',
                         'fmtint', 'fmtfloat', 'fmtdatetime']

        for pdtype in pandas_dtypes:

            try:

                inp_dict_cols[pdtype]

                if pdtype == 'key':
                    self.key = inp_dict_cols[pdtype].copy()

                if pdtype == 'fmtcategory':
                    self.fmtcategory = inp_dict_cols[pdtype]

                if pdtype == 'fmtordcategory':
                    self.fmtordcategory = inp_dict_cols[pdtype]

                if pdtype == 'fmtint':
                    self.fmtint = inp_dict_cols[pdtype]

                if pdtype == 'fmtfloat':
                    self.fmtfloat = inp_dict_cols[pdtype]

                if pdtype == 'fmtdatetime':
                    self.fmtdatetime = inp_dict_cols[pdtype]

            except Exception:
                continue

        dict_cols = dict()
        for key, value in inp_dict_cols.items():

            if key in pandas_dtypes + ['lstYtarget']:

                dict_cols[key] = value

        self.dict_cols = dict_cols.copy()

        # Outcome Message
        print('\n')
        print('Dataframe and column lists have been ingested')

# =============================================================================
# # <----- ----->
# # Exploratory Data Analysis
# # <----- ----->
# =============================================================================

    # Inner Function option settings
    def _options_plot_template(self, figsize=(22, 14),
                               palette='deep', style='white',
                               divpalette=sns.diverging_palette(240, 10, n=18),
                               left=0.05, right=0.95,
                               bottom=0.05, top=0.9,
                               wspace=0.15, hspace=0.2, fsize=12,
                               row=2, col=2):
        """
         Inner Function for setting the base template options for the
         EDA plot functions .

         Parameters
         _ _ _ _ _
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette name to display
         divpalette : string
             Seaborn diverging palette name to display (for heatmap only)
         style : string
             Seaborn style name to display
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         wspace : float
             Width space among the four subplots (horizontal)
         hspace : float
             High space among the four subplots (vertical)
        fsize : integer
             Font size of text plots

         Returns
         _ _ _ _ _
         Tuple of display options
             A subset of matplotilb parameters to be used by plot functions
        """
        sns.set(color_codes=True, style=style, palette=palette)

        cpalette = sns.color_palette()
        color = cpalette[3]

        fig = plt.figure(figsize=figsize, dpi=200)

        gs = GridSpec(row, col, figure=fig, left=left, right=right,
                      bottom=bottom, top=top, wspace=wspace, hspace=hspace,
                      width_ratios=None, height_ratios=None)

        set_suptitle = {'weight': 800, 'fontstyle': 'normal', 'size': fsize,
                        'stretch': 1000, 'horizontalalignment': 'center'}

        set_plttitle = {'weight': 800, 'fontstyle': 'oblique', 'size': fsize+2,
                        'stretch': 1000, 'horizontalalignment': 'center',
                        'color': color}

        set_subtitle = {'weight': 800, 'fontstyle': 'normal', 'size': fsize+1,
                        'stretch': 1000, 'horizontalalignment': 'left'}

        set_main = {'weight': 500, 'fontstyle': 'normal', 'size': fsize,
                    'stretch': 1000, 'horizontalalignment': 'right'}

        set_sub = {'weight': 500, 'fontstyle': 'normal', 'size': fsize-1,
                   'horizontalalignment': 'right'}

        return (fig, gs, set_suptitle, set_plttitle, set_subtitle,
                set_main, set_sub, cpalette, divpalette)

    # Inner Function for PiePlot
    def _fpieoverlap(self, text):
        """
         Inner Function for handling the pie plot visualization, so that
         the pie values do not overlap.
        """
        lstindex = [index for index, item in enumerate(text[2])
                    if float(item.get_text().strip('%')) / 100 < 0.01]

        lsttodel2 = [item for index, item in enumerate(text[2])
                     if index in lstindex]

        lsttodel1 = [item for index, item in enumerate(text[1])
                     if index in lstindex]

        lsttodelc = [lsttodel2, lsttodel1]

        lst2 = [item for index, item in enumerate(text[2])
                if index not in lstindex]

        lst1 = [item for index, item in enumerate(text[1])
                if index not in lstindex]

        lstc = [lst2, lst1]

        for i in lsttodelc:

            positions = [(round(item.get_position()[1], 1),
                          item) for item in i]

            textObjects = [i for position, i in positions]

            if textObjects:

                for textObject in textObjects:
                    textObject.set_visible(False)

        for i in lstc:

            positions = [(round(item.get_position()[1], 1),
                          item) for item in i]

            overLapping = Counter((item[0] for item in positions))

            overLapping = [key for key, value in overLapping.items()
                           if value >= 2]

            for key in overLapping:
                textObjects = [i for position, i in positions
                               if position == key]

                if textObjects:

                    for textObject in textObjects:
                        textObject.set_visible(False)

    # Inner Function to prepare numerical field metrics
    def _series_numplot_metrics(self, field):
        """
         Inner Function for computing numerical metrics used by EDA plots.

         Parameters
         _ _ _ _ _
         field : Pandas Series
             The numerical Pandas Series wich the metrics are computed on

         Returns
         _ _ _ _ _
         Dictionary
             The dictionary of key: metrics name, value: metrics value
        """
        dict_metrics = dict()
        print(field)
        dict_metrics['mean_TOT'] = round(self.df[field].mean(), 2)
        dict_metrics['std_TOT'] = round(self.df[field].std(), 2)

        dict_metrics['mean_0'] = round(self.df[self.yvals == 0][
                 field].mean(), 2)
        dict_metrics['mean_1'] = round(self.df[self.yvals == 1][
                 field].mean(), 2)

        dict_metrics['std_0'] = round(self.df[self.yvals == 0][
                 field].std(), 2)
        dict_metrics['std_1'] = round(self.df[self.yvals == 1][
                 field].std(), 2)

        if not self.df[field].isna().any():

            dict_metrics['fvalue'] = np.round(f_classif(self.df[[field]],
                                              self.yvals)[0], 2)
            dict_metrics['pvalue'] = np.round(f_classif(self.df[[field]],
                                              self.yvals)[1], 5)

            dict_metrics['mutual_info'] = np.round(mutual_info_classif(
                    self.df[[field]], self.yvals,
                    discrete_features='auto', n_neighbors=3), 5)

        dict_metrics['outlier_plus'] = self.df[
                self.df[field] > (
                        dict_metrics['mean_TOT']
                        + dict_metrics['std_TOT'] * 3)][field].count()

        dict_metrics['outlier_minus'] = self.df[
                self.df[field] < (
                        dict_metrics['mean_TOT']
                        - dict_metrics['std_TOT'] * 3)][field].count()

        dict_metrics['nan_values'] = int(self.df[field].isna().sum())
        dict_metrics['values'] = int(self.df[field].notna().sum())

        return dict_metrics

    # Inner Function to prepare categorical field metrics
    def _series_catplot_metrics(self, field, ctype='low'):
        """
         Inner Function for computing categorical metrics used by EDA plots.

         Parameters
         _ _ _ _ _
         field : Pandas Series
             The numerical Pandas Series which the metrics are computed on

         Returns
         _ _ _ _ _
         Dictionary
             The dictionary of key: metrics name, value: metrics value
        """
        dict_metrics = dict()

        df = self.df[[field]]
        dfenc = ce.OrdinalEncoder().fit_transform(df)

        dict_metrics['nlevels'] = len(df[field].unique())

        if not df[field].isna().any():

            dict_metrics['chi2value'] = np.round(chi2(dfenc,
                                                 self.yvals)[0], 2)

            dict_metrics['chi2pvalue'] = np.round(chi2(dfenc,
                                                  self.yvals)[1], 5)

            dict_metrics['mutual_info'] = np.round(mutual_info_classif(
                    dfenc, self.yvals,
                    discrete_features=True, n_neighbors=3), 5)

        dict_metrics['nan_values'] = int(df[field].isna().sum())
        dict_metrics['values'] = int(df[field].notna().sum())

        dict_metrics['df2cross'] = pd.concat(
                [df.reset_index(drop=True),
                 self.yvals.reset_index(drop=True)],
                axis=1, join='inner').set_index(df.index)

        dict_metrics['dfcorr_STD'] = pd.crosstab(
                self.yvals,
                dict_metrics['df2cross'][field])

        dict_metrics['dfcorr_NORM'] = pd.crosstab(
                self.yvals,
                dict_metrics['df2cross'][field],
                normalize='columns')

        if ctype == 'high':

            dict_metrics['field_NORM'] = pd.Series(data=dict_metrics[
                    'dfcorr_NORM'].loc[1], name='NORM').sort_values()

            field_VALUES = dict_metrics[
                    'df2cross'][field].value_counts().rename('VALUES')

            dict_metrics['field_VALUES'] = field_VALUES[:20]

            df_field = pd.concat([field_VALUES, dict_metrics['field_NORM']],
                                 join='inner', axis=1).sort_values(
                                         'VALUES',
                                         ascending=False).iloc[:20, :]

            dict_metrics['df_LARGE'] = df_field.iloc[:10, :].sort_values(
                    'VALUES', ascending=False)

            dict_metrics['df_SMALL'] = df_field.iloc[-10:, :].sort_values(
                    'VALUES', ascending=False)

        return dict_metrics

    # Inner Function numerical option settings
    def _prepare_numplot_template(self, ax0, dictm, field,
                                  options):
        """
         Inner Function for preparing the numerical plot template.

         Parameters
         _ _ _ _ _
         ax0 : Matplotlib axes
             It is the first subplot (axes) of EDA plots
         dictm : Dictionary
             dictionary of metrics (returned by self.*_metrics() function)
         field : Pandas Series
             The Pandas Series wich the template is to build on
         options : Display options (returned by self._options_plot_template())
             A sub set of matplotlib and seaborn options

         Returns
         _ _ _ _ _
         Matplotlib axes setting
             It is the first subplot (axes) of EDA plots
        """
        fig, gs, set_suptitle, set_plttitle, set_subtitle, \
            set_main, set_sub, cpalette, divpalette = options

        ax0[0].get_xaxis().set_ticks([])
        ax0[0].get_yaxis().set_ticks([])

        # Summary Statistics
        ax0[0].text(0.05, 0.90, 'Summary Statistics ', set_subtitle)

        ax0[0].text(0.27, 0.80, 'Total Series ', set_main)
        ax0[0].text(0.55, 0.80,
                    'Mean :  ' + str(dictm['mean_TOT']), set_sub)
        ax0[0].text(0.90, 0.80,
                    'Std :  ' + str(dictm['std_TOT']), set_sub)

        ax0[0].text(0.27, 0.70, 'The 0 Event ', set_main)
        ax0[0].text(0.55, 0.70,
                    'Mean :  ' + str(dictm['mean_0']), set_sub)
        ax0[0].text(0.90, 0.70,
                    'Std :  ' + str(dictm['std_0']), set_sub)

        ax0[0].text(0.27, 0.60, 'The 1 Event ', set_main)
        ax0[0].text(0.55, 0.60,
                    'Mean :  ' + str(dictm['mean_1']), set_sub)
        ax0[0].text(0.90, 0.60,
                    'Std :  ' + str(dictm['std_1']), set_sub)

        ax0[0].text(0.27, 0.50, 'Outlier : ', set_main)
        ax0[0].text(0.55, 0.50,
                    'Num Pos :  ' + str(dictm['outlier_plus']), set_sub)
        ax0[0].text(0.90, 0.50,
                    'Num Neg :  ' + str(dictm['outlier_minus']), set_sub)

        ax0[0].text(0.27, 0.40, 'Values : ', set_main)
        ax0[0].text(0.55, 0.40,
                    'Num Nan :  ' + '{:0,}'.format(dictm['nan_values']),
                    set_sub)

        ax0[0].text(0.90, 0.40,
                    'Num Val :  ' + '{:0,}'.format(dictm['values']),
                    set_sub)

        # Statistical Testing
        if not self.df[field].isna().any():

            ax0[0].text(0.05, 0.30, 'Test Statistics ', set_subtitle)

            ax0[0].text(0.27, 0.20, 'Anova Test ', set_main)
            ax0[0].text(0.55, 0.20,
                        'Fvalue : ' + str(dictm['fvalue']), set_sub)
            ax0[0].text(0.90, 0.20,
                        'pvalue : ' + str(dictm['pvalue']), set_sub)

            ax0[0].text(0.27, 0.10, 'Mutual Info ',  set_main)
            ax0[0].text(0.90, 0.10,
                        'Statistical Value : ' + str(dictm['mutual_info']),
                        set_sub)
        else:

            ax0[0].text(
                    0.05, 0.20,
                    'The Field shows nan values -  No Test Statics plotted')

        return ax0[0]

    # Inner Function categorical option settings
    def _prepare_catplot_template(self, ax0, dictm, field,
                                  options):
        """
         Inner Function for preparing the categorical plot template.

         Parameters
         _ _ _ _ _
         ax0 : Matplotlib axes
             It is the first subplot (axes) of EDA plots
         dictm : Dictionary
             dictionary of metrics (returned by self.*_metrics() function)
         field : Pandas Series
             The Pandas Series wich the template is to build on
         options : Display options (returned by self._options_plot_template())
             A sub set of matplotlib and seaborn options

         Returns
         _ _ _ _ _
         Matplotlib axes setting
             It is the first subplot (axes) of EDA plots
        """
        fig, gs, set_suptitle, set_plttitle, set_subtitle, \
            set_main, set_sub, cpalette, divpalette = options

        # Summary Statistics
        ax0[0].text(0.05, 0.90, 'Summary Statistics ', set_subtitle)
        ax0[0].get_xaxis().set_ticks([])
        ax0[0].get_yaxis().set_ticks([])

        # Mean and Std Metrics
        ax0[0].text(0.27, 0.80, 'Cardinality ', set_main)
        ax0[0].text(0.55, 0.80,
                    'Num Levels :  ' + str(dictm['nlevels']), set_sub)
        ax0[0].text(0.90, 0.80,
                    'Avg Records :  ' + '{:0,.1f}'.format(round(
                            dictm['values'] / dictm['nlevels'], 1)),
                    set_sub)

        ax0[0].text(0.27, 0.70, 'Values ', set_main)
        ax0[0].text(0.55, 0.70,
                    'Num Nan :  ' + '{:0,}'.format(dictm['nan_values']),
                    set_sub)
        ax0[0].text(0.90, 0.70,
                    'Num Val :  ' + '{:0,}'.format(dictm['values']),
                    set_sub)

        # Statistical Testing
        if not self.df[field].isna().any():

            ax0[0].text(0.05, 0.50, 'Test Statistics ', set_subtitle)

            ax0[0].text(0.27, 0.40, 'Chi2 Test ', set_main)
            ax0[0].text(0.55, 0.40,
                        'value : ' + str(dictm['chi2value']), set_sub)
            ax0[0].text(0.90, 0.40,
                        'pvalue : ' + str(dictm['chi2pvalue']), set_sub)

            ax0[0].text(0.27, 0.30, 'Mutual Info ',  set_main)
            ax0[0].text(0.90, 0.30,
                        'Statistical Value : ' + str(dictm['mutual_info']),
                        set_sub)
        else:

            ax0[0].text(
                    0.05, 0.20,
                    'The Field shows nan values - No Test Statics plotted')

        return ax0[0]

    # Numerical Series To Binary Target Correlation Plot Template
    def exploratory_numerical_plot(self, field, root, path,
                                   smoothe_outlier=False,
                                   figsize=(22, 14),
                                   palette='deep', style='white',
                                   left=0.07, right=0.95,
                                   bottom=0.05, top=0.9,
                                   wspace=0.18, hspace=0.22, fsize=12,
                                   setslot=None):
        """
         EDA numerical plot template.
         The function generates an exploratory data visualization figure for
         the numerical provided column, splitted in four subplots.
         In the top-left one are plotted summary and test statistics.
         For the latter are used a sklearn f-classif (one way anova) and a
         sklearn mutual information classif.
         In the top-right is located the hist-kde feature distribution,
         (seaborn distplot).
         In the bottom left-right, rispectevely, a stripplot and a boxen plot
         showing the feature distribution across the two target values, for
         Event0 and Event1.

         Parameters
         _ _ _ _ _
         field : Pandas Series
             The numerical Pandas Series which the metrics are computed on
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         smoothe_outlier: Boolean
             If to smoothe outlyer values for the strip plot and boxen plot by
             replacing outlier values with the mean+3std one, or the
             mean-3std for negative values.
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette to display
         style : string
             Seaborn style to display
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         wspace : float
             Width space among the four subplots (horizontal)
         hspace : float
             High space among the four subplots (vertical)
        fsize : integer
             Font size of text plots
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.exploratory_numerical_plot,
                                    dict_locals,
                                    setslot)
            return

        def fpctfmt(x, pos):
            pctfmt = '{:,.2f}%'.format(x * 100.0)
            return pctfmt

        def fthousfmt(x, pos):
            thousfmt = '{:,d}'.format(int(x // 100))
            return thousfmt

        # pctfmt = FuncFormatter(fpctfmt)
        thousfmt = FuncFormatter(fthousfmt)

        dictm = self._series_numplot_metrics(field)

        options = self._options_plot_template(figsize=figsize,
                                              palette=palette, style=style,
                                              left=left, right=right,
                                              bottom=bottom, top=top,
                                              wspace=wspace, hspace=hspace,
                                              fsize=fsize)

        fig, gs, set_suptitle, set_plttitle, set_subtitle, \
            set_main, set_sub, cpalette, divpalette = options
        c_one = cpalette[0]

        # Plotting Grid Space
        ax0, ax1 = [0, 1], [0, 1]

        ax0[0] = fig.add_subplot(gs[0, 0])
        ax0[0].set_title(field.capitalize(), set_plttitle)

        ax0[1] = fig.add_subplot(gs[0, 1])
        ax0[1].set_title('Feature Univariate Distribution', set_plttitle)

        ax1[0] = fig.add_subplot(gs[1, 0])
        ax1[0].set_title('Strip on Binary Target', set_plttitle)

        ax1[1] = fig.add_subplot(gs[1, 1], sharey=ax1[0])
        ax1[1].set_title('Boxen on Binary Target', set_plttitle)

        fig.suptitle('Exploratory and Correlation Plots for Numericals',
                     weight=1000, fontstyle='oblique', size=20, stretch=1000,
                     color=c_one)

        # Axis ax0[0]
        ax0[0] = self._prepare_numplot_template(ax0, dictm, field, options)

        if smoothe_outlier:

            df = self.engineer_smooth_outlier(col_lists=[field],
                                              mode='col_names', multiplier=3)

        df = pd.concat([self.df[field].reset_index(drop=True),
                        self.yvals.reset_index(drop=True)],
                       axis=1, join='inner').set_index(self.df.index)

        # Axis ax0[1]
        sns.distplot(self.df[field].dropna(), ax=ax0[1])

        if df[field].max() > 1000:
            ax0[1].set_xlabel(xlabel='values in k', style='oblique')
            ax0[1].xaxis.set_major_formatter(thousfmt)

        else:
            ax0[1].set_xlabel(xlabel='')

        # Axis ax1[0]
        sns.stripplot(data=df.dropna(), y=field, x=self.y, ax=ax1[0])
        ax1[0].set_xlabel(xlabel='')

        def _yaxis(ax):
            if df[field].max() > 1000:
                ax.set_ylabel(ylabel='values in k', style='oblique')
                ax.yaxis.set_major_formatter(thousfmt)
            else:
                ax.set_ylabel('')

            return ax

        ax1[0] = _yaxis(ax1[0])

        # Axis ax1[1]
        sns.boxenplot(data=df.dropna(), y=field, x=self.y, ax=ax1[1])
        ax1[1].set_xlabel(xlabel='')

        ax1[1] = _yaxis(ax1[1])

        plt.savefig(root + path + '\\' + field + '.jpg', dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + field + '.jpg' +
              ' exported to ' + root + path)

    # Low Categorical Series To Binary Target Correlation Plot Template
    def exploratory_lowcat_plot(self, field, root, path, figsize=(22, 14),
                                palette='deep', style='white',
                                divpalette=sns.diverging_palette(
                                       240, 10, n=18),
                                left=0.05, right=0.95,
                                bottom=0.05, top=0.9,
                                wspace=0.15, hspace=0.2, fsize=12,
                                setslot=None):
        """
         EDA low cardinality categorical plot template.
         The function generates an exploratory data visualization figure for
         the low cardinality provided column, splitted in four subplots.
         In the top-left one are plotted summary and test statistics.
         For the latter are used a sklearn chi2 (one way anova) and a
         sklearn mutual information classif.
         In the top-right is located the feature pie-plot (metric: counts)
         In the bottom left-right are located two heatmaps, 1 standard (left)
         and 1 normalized (right), of the feature against the target variable.

         Parameters
         _ _ _ _ _
         field : Pandas Series
             The categorical Pandas Series which the metrics are computed on
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette to display
         divpalette : string
             Seaborn diverging palette name to display (for heatmap only)
         style : string
             Seaborn style to display
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         wspace : float
             Width space among the four subplots (horizontal)
         hspace : float
             High space among the four subplots (vertical)
        fsize : integer
             Font size of text plots
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.exploratory_lowcat_plot,
                                    dict_locals,
                                    setslot)
            return

        dictm = self._series_catplot_metrics(field)

        # Figure Plot
        options = self._options_plot_template(figsize=figsize,
                                              palette=palette, style=style,
                                              divpalette=divpalette,
                                              left=left, right=right,
                                              bottom=bottom, top=top,
                                              wspace=wspace, hspace=hspace,
                                              fsize=fsize)

        fig, gs, set_suptitle, set_plttitle, set_subtitle, \
            set_main, set_sub, cpalette, divpalette = options
        c_one = cpalette[0]

        # Plotting Grid Space
        ax0, ax1 = [0, 1], [0, 1]

        ax0[0] = fig.add_subplot(gs[0, 0])
        ax0[0].set_title(field.capitalize(), set_plttitle)
        ax0[1] = fig.add_subplot(gs[0, 1])
        ax0[1].set_title('Feature Pie Plot', set_plttitle)
        ax1[0] = fig.add_subplot(gs[1, 0])
        ax1[0].set_title('Heatmap on Binary Target', set_plttitle)
        ax1[1] = fig.add_subplot(gs[1, 1])
        ax1[1].set_title('Heatmap on Binary Target (Normalized)', set_plttitle)

        fig.suptitle('Exploratory and Correlation Plots for low categoricals',
                     weight=1000, fontstyle='oblique',
                     size=20, stretch=1000, color=c_one)

        # Axis ax0[0]
        ax0[0] = self._prepare_catplot_template(ax0, dictm, field,
                                                options)
        # Axis ax0[1]
        values = self.df[field].value_counts(normalize=True).values
        index = self.df[field].value_counts(normalize=True).index

        piechart = ax0[1].pie(values, labels=index,
                              autopct='%1.1f%%')

        self._fpieoverlap(piechart)

        # Axis ax1[0]
        annot_kws = {'ha': 'center', 'va': 'center', 'size': 14}

        sns.heatmap(dictm['dfcorr_STD'], annot=True, annot_kws=annot_kws,
                    cbar=False, linewidths=.5, fmt=',',
                    cmap=divpalette, ax=ax1[0])

        def heat_set(ax):
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom + 0.5, top - 0.5)
            ax.set_ylabel('')
            ax.set_xlabel('')

        heat_set(ax1[0])

        # Axis ax1[1]
        sns.heatmap(dictm['dfcorr_NORM'], annot=True, annot_kws=annot_kws,
                    cbar=False, linewidths=.5, fmt='.1%',
                    cmap=divpalette, ax=ax1[1])

        heat_set(ax1[1])

        plt.savefig(root + path + '\\' + field + '.jpg', dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + field + '.jpg' +
              ' exported to ' + root + path)

    # High Categorical Series To Binary Target Correlation Plot Template
    def exploratory_highcat_plot(self, field, root, path,
                                 figsize=(22, 14),
                                 palette='deep', style='white',
                                 left=0.08, right=0.95,
                                 bottom=0.2, top=0.9,
                                 wspace=0.2, hspace=0.2, fsize=12,
                                 setslot=None):
        """
         EDA high cardinality categorical plot template.
         The function generates an exploratory data visualization figure for
         the high cardinality provided column, splitted in four subplots.
         In the top-left one are plotted summary and test statistics.
         For the latter are used a sklearn chi2 (one way anova) and a
         sklearn mutual information classif.
         In the top-right is located the distribution of the Target Encoding
         (Event 1 percentage) for the categorical feature (seaborn distplot).
         Indeed, the plot is computed on the top of the feature target
         encoding, which returns the results of the following formula :
         Event1 / (Event1 + Event0) - computed For each feature level
         In the bottom left-right suplots are located two combined plots,
         showing the feature levels with the highest Event 1 % mean,
         the left one, and the smallest, the right one.
         The Event 1 Target % is represented by the dots (y right axes)
         The level counts (num records) are showned by the bars (y left axes)

         Parameters
         _ _ _ _ _
         field : Pandas Series
             The categorical Pandas Series which the metrics are computed on
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette to display
         divpalette : string
             Seaborn diverging palette name to display (for heatmap only)
         style : string
             Seaborn style to display
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         wspace : float
             Width space among the four subplots (horizontal)
         hspace : float
             High space among the four subplots (vertical)
        fsize : integer
             Font size of text plots
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.exploratory_highcat_plot,
                                    dict_locals,
                                    setslot)
            return

        def fpctfmt(x, pos):
            pctfmt = '{:,.0f}%'.format(x * 100.0)
            return pctfmt

        def fthousfmt(x, pos):
            thousfmt = '{:,d}'.format(int(x // 100))
            return thousfmt

        pctfmt = FuncFormatter(fpctfmt)
        thousfmt = FuncFormatter(fthousfmt)

        dictm = self._series_catplot_metrics(field, ctype='high')

        # Figure Plot
        options = self._options_plot_template(figsize=figsize,
                                              palette=palette, style=style,
                                              left=left, right=right,
                                              bottom=bottom, top=top,
                                              wspace=wspace, hspace=hspace,
                                              fsize=fsize)

        fig, gs, set_suptitle, set_plttitle, set_subtitle, \
            set_main, set_sub, cpalette, divpalette = options
        c_one, c_two = cpalette[0], cpalette[5]

        # Plotting Grid Space
        ax0, ax1 = [0, 1], [0, 1]

        ax0[0] = fig.add_subplot(gs[0, 0])
        ax0[0].set_title(field.capitalize(), set_plttitle)

        ax0[1] = fig.add_subplot(gs[0, 1])
        ax0[1].set_title('Feature Distribution on Event1 %',
                         set_plttitle)
        ax1[0] = fig.add_subplot(gs[1, 0])
        ax1[0].set_title('Plot of nlargest Event1 %',
                         set_plttitle)
        ax1[1] = fig.add_subplot(gs[1, 1], sharey=ax1[0])
        ax1[1].set_title('Plot of nsmallest Event1 %',
                         set_plttitle)

        fig.suptitle('Exploratory and Correlation Plots for high categoricals',
                     weight=1000, fontstyle='oblique',
                     size=20, stretch=1000, color=c_one)

        # Axis ax0[0]
        ax0[0] = self._prepare_catplot_template(ax0, dictm, field,
                                                options)

        # Axis ax0[1]
        encoder = ce.TargetEncoder()

        dfencoded = encoder.fit_transform(dictm['df2cross'][field],
                                          self.yvals)

        sns.distplot(dfencoded.dropna(), ax=ax0[1],
                     kde_kws={'gridsize': 20})

        # Axis ax1[0]
        def _combined_plot(df, ax):

            ax.bar(df.index, df['VALUES'], color=sns.color_palette())
            ax.set_ylabel('Number of Records')

            def _yaxis(ax):
                if self.df[field].value_counts().mean() > 1000:
                    ax.set_ylabel(ylabel='values in k', style='oblique')
                    ax.yaxis.set_major_formatter(thousfmt)
                else:
                    ax.set_ylabel('')

                return ax

            ax = _yaxis(ax)

            ax_2 = ax.twinx()

            ax_2.plot(df.index, df['NORM'],
                      marker='>', linestyle='-',
                      color=c_two)

            ax_2.set_ybound(upper=1.1, lower=-0.1)
            ax_2.set_yticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1],
                                 color=c_two)
            ax_2.yaxis.set_major_formatter(pctfmt)

            ax.set_xticklabels(df.index, rotation='vertical')
            ax.set_xlabel('')

        _combined_plot(dictm['df_LARGE'], ax1[0])

        # Axis ax1[1]
        _combined_plot(dictm['df_SMALL'], ax1[1])
        ax1[1].set_ylabel('')
        # ax1[1].get_yaxis().set_ticks([])

        plt.savefig(root + path + '\\' + field + '.jpg', dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + field + '.jpg' +
              ' exported to ' + root + path)

    # Categorical Df Columns To Binary Target Correlation Plot Loop
    def exploratory_df_lowcat_plots(self, root, path,
                                    figsize=(22, 14),
                                    palette='deep', style='white',
                                    divpalette=sns.diverging_palette(
                                       240, 10, n=18),
                                    left=0.05, right=0.95,
                                    bottom=0.05, top=0.9,
                                    wspace=0.15, hspace=0.2, fsize=18,
                                    setslot=None):
        """
         EDA dataframe low categorical figures export.
         By looping over the exploratory_lowcat_plot, the function exports
         a matplotlib figure for each of the low cardinality categorical
         variables located in the data container to a provided path.
         For the template building details check exploratory_lowcat_plot.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         options : Matplotlib and seaborn settings
             The same as exploratory_lowcat_plot function parameters
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figures
             n matplotilb figures for n low categorical Fastlane vars
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.exploratory_df_lowcat_plots,
                                    dict_locals,
                                    setslot)
            return

        categoricals = self.fmtcat_low

        for col in tqdm(self.df[categoricals].columns):

            self.exploratory_lowcat_plot(col, root, path,
                                         figsize=figsize,
                                         palette=palette, style=style,
                                         divpalette=divpalette,
                                         left=left, right=right,
                                         bottom=bottom, top=top,
                                         wspace=wspace, hspace=hspace,
                                         fsize=fsize,
                                         setslot=setslot)

    # Categorical Df Columns To Binary Target Correlation Plot Loop
    def exploratory_df_highcat_plots(self, root, path,
                                     figsize=(22, 14),
                                     palette='deep', style='white',
                                     left=0.08, right=0.95,
                                     bottom=0.2, top=0.9,
                                     wspace=0.2, hspace=0.2, fsize=18,
                                     setslot=None):
        """
         EDA dataframe high categorical figures export.
         By looping over the exploratory_highcat_plot, the function exports
         a matplotlib figure for each of the high cardinality categorical
         variables located in the data container to a provided path.
         For the template building details check exploratory_highcat_plot.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         options : Matplotlib and seaborn settings
             The same as exploratory_highcat_plot function parameters
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figures
             n matplotilb figures for n high categorical Fastlane vars
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.df_highcatcols_corr_plots,
                                    dict_locals,
                                    setslot)
            return

        categoricals = self.fmtcat_high

        for col in tqdm(self.df[categoricals].columns):

            self.exploratory_highcat_plot(col, root, path,
                                          figsize=figsize,
                                          palette=palette, style=style,
                                          left=left, right=right,
                                          bottom=bottom, top=top,
                                          wspace=wspace, hspace=hspace,
                                          fsize=fsize,
                                          setslot=setslot)

    # Numerical Df Columns To Binary Target Correlation Plot Loop
    def exploratory_df_numerical_plots(self, root, path,
                                       smoothe_outlier=False,
                                       figsize=(22, 14),
                                       palette='deep', style='white',
                                       left=0.07, right=0.95,
                                       bottom=0.05, top=0.9,
                                       wspace=0.18, hspace=0.22, fsize=18,
                                       setslot=None):
        """
         EDA dataframe numerical figures export.

         By looping over the exploratory_numerical_plot, the function exports
         a matplotlib figure for each of the numerical variables located in
         the data container to a provided path.
         For the template building details check exploratory_numerical_plot
         function.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         smoothe_outlier: Boolean
             If to smoothe outlyer values for the strip plot and boxen plot by
             replacing outlier values with the mean+3std one, or the
             mean-3std for negative values.
         options : Matplotlib and seaborn settings
             The same as exploratory_numerical_plot function parameters
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figures
             n matplotilb figures for n numerical Fastlane variables
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.exploratory_df_numerical_plots,
                                    dict_locals,
                                    setslot)
            return

        numericals = self.fmtfloat + self.fmtint

        for col in tqdm(self.df[numericals].columns):

            self.exploratory_numerical_plot(col, root, path,
                                            smoothe_outlier=smoothe_outlier,
                                            figsize=figsize,
                                            palette=palette, style=style,
                                            left=left, right=right,
                                            bottom=bottom, top=top,
                                            wspace=wspace, hspace=hspace,
                                            fsize=fsize,
                                            setslot=setslot)

    # Pandas Profiling
    def exploratory_pandas_profiling(self, root, path,
                                     filename='ExploratoryDataAnalysis.html',
                                     setslot=None):
        """
        The methods launchs the pandas profiling function on the dataframe
        ingested into the BinaryLane.
        If the dataframe has already been splitted between train and test
        the profiling is applied only to the train test.
        If it has also been balanced, it is applied on the balanced one.
        Generates profile reports from a pandas DataFrame.

        Pandas_profiling extends the pandas DataFrame with df.profile_report()
        for quick data analysis.

        For each column the following statistics, re presented in an
        interactive HTML report:

        Type inference: detect the types of columns in a dataframe.
        Essentials: type, unique values, missing values
        Quantile statistics like minimum value, Q1, median, Q3, maximum,
        range, interquartile range
        Descriptive statistics like mean, mode, standard deviation, sum,
        median absolute deviation, coefficient of variation, kurtosis, skewness
        Most frequent values
        Histogram
        Correlations highlighting of highly correlated variables, Spearman,
        Pearson and Kendall matrices
        Missing values matrix, count, heatmap and dendrogram of missing values
        Text analysis learn about categories (Uppercase, Space),
        scripts (Latin, Cyrillic) and blocks (ASCII) of text data.
        File and Image analysis extract file sizes, creation dates and
        dimensions and scan for truncated images or those containing
        XIF information.

        link : https://github.com/pandas-profiling/pandas-profiling

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         filename: String
            The filename the figure is exported to
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Pandas Profiling Report
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.pandas_profiling,
                                    dict_locals,
                                    setslot)
            return

        path = root + path
        profile = pp.ProfileReport(self.df)
        profile.to_file(outputfile=path + '\\' + filename)

        print('Pd_Profiling_Report' + ' exported in ' + path)

    # Plot model confusion matrix
    def exploratory_correlation_plot(self, root, path,
                                     col_lists=None, mode='all',
                                     figsize=(20, 12), bottom=0.2, top=0.95,
                                     divpalette=sns.diverging_palette(
                                             240, 10, n=18),
                                     setslot=None):
        """
         Display an Heatmap on the basis of a pearson computed pairwise columns
         correlation. The columns provided need to be numericals, including
         the categorical ones that must have been previously encoded.
         The method is strictly connected to the columns_drop_correlated one,
         which enable the user to drop the columns that showed to have a
         correlation index higher than a certain threshold.
         Three modes are available for passing the columns to the method :
         With the first one 'all', default, the method will be applied
         on the entire dataframe .
         The second mode, 'fmt_names', enables to work with pandas data
         type group of columns. In such a case the col_lists param must be
         imputed with one or more of the following (in list).

                 'fmtcategory' : categorical columns
                 'fmtordcategory' : ordinal categorical columns
                 'fmtint' : integer number columns
                 'fmtfloat' : float number columns
                 'fmtdatetime' : date and datetime columns

         In the 'col_names' mode, instead, the single given colums (in list)
         are used for imputation.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         col_lists : list
             List of pandas data type labels or single columns
         mode : 'string'
             'all' : all of the dataframe columns
             'col_names' : if single columns are given
             'fmt_names : if pandas data types group labels are given
         figsize : tuple
             To set Matplotilib figure size
         bottom : float
             Bottom Figure Margin
         top : float
             Top Figure Margin
         divpalette : seaborn palette
             All the seaborn diverging palette are accepted
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.exploratory_correlation_plot,
                                    dict_locals,
                                    setslot)
            return

        columns = self._apply_columns(col_lists, mode=mode)

        filename = 'Correlation Matrix Plot'

        fig, ax = plt.subplots(figsize=figsize)

        fig.suptitle('Correlation Matrix Plot',
                     weight=1000, fontstyle='oblique',
                     size=20, stretch=1000)

        matrix = self.df[columns].corr()

        npmask = np.zeros_like(matrix, dtype=np.bool)

        npmask[np.triu_indices_from(npmask)] = True

        sns.heatmap(matrix, square=True, annot=False,
                    linewidths=.5, mask=npmask,
                    cmap=divpalette, ax=ax)

        bottomax, topax = ax.get_ylim()
        ax.set_ylim(bottomax + 0.5, topax - 0.5)

        plt.subplots_adjust(bottom=bottom, top=top)

        plt.savefig(root + path + '\\' + filename + '.jpg', dpi=300)

        fig.clear()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + filename + '.jpg' +
              ' exported to ' + root + path)

# =============================================================================
# # <----- ----->
# # Pre Processing
# # <----- ----->
# =============================================================================

    # Inner Function - drop columns from lists and dictionary
    def _drop_col(self, col_name, drop_df=True):
        """
         Inner method of the class for dropping a colum name from all of the
         inner object attributes, such as lists or dictionaries.
         If drop_df=True it also removes the column from dataframe
         object instances.

         By calling the method at the moment of df column deletion is possible
         to keep aligned every component of the class, avoiding unexpected
         results when calling other methods.

         Parameters
         _ _ _ _ _
         col_name : string
        """

        message = ' column has been removed'

        selector_list = ['_univariate_selcat',
                         '_univariate_selnum',
                         '_model_selkfold',
                         '_boruta_selkfold',
                         '_boruta_select_stats',
                         '_model_select_stats',
                         '_explained_rfs']

        # Drop col from dict_cols list
        for key, columns in self.dict_cols.items():

            if col_name in columns:

                self.dict_cols[key].remove(col_name)

                # Outcome Message
                print(str(col_name) + message)

        # Drop col from derived list attributes
        obj_list_cols = []

        for key, values in self.__dict__.items():

            if key[0:3] == 'fmt' or key[0:5] == '_cols' \
                    or key in selector_list:

                obj_list_cols.append(values)

        for obj_list in obj_list_cols:

            if type(obj_list) == dict:

                for k, v in obj_list.items():

                    if col_name == k:
                        del obj_list[col_name]
                        break

                    if col_name in v:
                        obj_list[k].remove(col_name)
                        break

            elif type(obj_list) == list:

                if col_name in obj_list:

                    obj_list.remove(col_name)

            elif type(obj_list) == pd.DataFrame:

                if col_name in obj_list:

                    obj_list.drop(col_name, inplace=True)

        # Drop col from dataframes
        if drop_df:

            for key, df in self.__dict__.items():

                if type(df) == pd.DataFrame and col_name in df.columns:

                    df = df.drop(columns=[col_name]).copy()
                    setattr(self, key, df)

    # Inner Function - Encoding cat vars in a kfold supervised fashion
    def _kfold_encoding(self, encoder, n_splits=5, apply2test=False):
        """
         Inner method of the class for supervised encoding in a k fold fashion.

         Parameters
         _ _ _ _ _
         encoder : Category Encoder istance
             Expected an istance of one of the classes provided by the
             category_encoders Library (only supervised are accepted).
             Example : ce.TargetEncoder()
             link : http://contrib.scikit-learn.org/category_encoders/
         n_splits : Integer
             The number of kfold the encoder estimator will be evaluated on
             sklearn.model_selection Kfold class is used for splitting
         apply2test : Boolean
             If to apply the category encoder transform method on the test set

         Returns
         _ _ _ _ _
         Encoded DataFrame
        """
        kfold = KFold(n_splits)
        gen_kfold = kfold.split(self.df)
        df_enc = pd.DataFrame()

        for train, test in gen_kfold:

            train = encoder.fit(self.df.iloc[train],
                                self.yvals[train])

            test = encoder.transform(self.df.iloc[test])

            df_enc = df_enc.append(test)

        df_enc.sort_index(inplace=True)

        if apply2test:

            df_orig_col2enc = self.df[encoder.get_params()["cols"]]

            for col in df_orig_col2enc.columns:

                col_enc = col + "_encoded"

                df_join = pd.DataFrame(zip(df_orig_col2enc[col],
                                           df_enc[col]),
                                       columns=[col, col_enc])

                groupby = df_join.groupby(col)[col_enc].mean()
                mapping = dict(groupby)

                new_vals = [x for x in self.dftest[col].unique()
                            if x not in mapping.keys()]

                if new_vals:

                    # Outcome Message
                    msg = 'has shown ' + str(len(new_vals)) + \
                        ' Unknown values in the test set'

                    print('\n')
                    print(col, msg, new_vals)

                    self.dftest.loc[self.dftest[col].isin(
                            new_vals), col] = df_enc[col].mean()

                self.dftest[col] = self.dftest[col].replace(mapping)

            return (df_enc, self.dftest)

        return df_enc

    # Inner Function - Encoding intenal function
    def _set_encode(self, encoder, columns, enc_type='standard',
                    n_splits=5,
                    apply2test=False):
        """
         Inner method of the class for managing two encoding types :
             'standard' : simple category encoder fit_transform method
             'kfold' : cross validation encoding

         Parameters
         _ _ _ _ _
         encoder : Category Encoder istance
             Expected an istance of one of the classes provided by the
             category_encoders Library.
         columns : List
             List of categorical columns the encoder must be applied on
         enc_type : String
             Two possible values : 'standard', 'kfold'
         n_splits : Integer
             The number of kfold the encoder estimator will be evaluated on
         apply2test : Boolean
             If to apply category encoder transform method on the test set

         Returns
         _ _ _ _ _
         Encoded DataFrame
             self instance setting : df
        """
        encoder.set_params(cols=columns)

        if enc_type == 'standard':

            self.df = encoder.fit_transform(self.df,
                                            self.yvals).sort_index().copy()

            if apply2test:

                self.dftest = encoder.transform(
                        self.dftest).sort_index().copy()

        elif enc_type == 'kfold':

            if not apply2test:

                self.df = self._kfold_encoding(
                        encoder, n_splits, apply2test).sort_index().copy()

            elif apply2test:

                self.df, self.dftest = self._kfold_encoding(
                        encoder, n_splits, apply2test)

                self.df.sort_index(inplace=True)
                self.dftest.sort_index(inplace=True)

    # Inner Function - Handling One Hot Encoding Exception
    def _get_onehotenc_dictcols(self, cols_orig):
        """
         Inner method of the class for handling one hot encoded fashion
         encoders, with reference to column names.
         By calling the method, the mapping between the base column name,
         used in the ohe method, and the derived-encoded columns 1:n
         (levels of cardianlity) is returned.

         Parameters
         _ _ _ _ _
         cols_orig : List
             List of colum names which the one hot encoded columns are
             generated from

         Returns
         _ _ _ _ _
         dict_mapp : Dictionary
             Dictionary with key : base column, values : ohe columns
        """
        cols_dif = [x for x in self.df.columns if x not in cols_orig]

        if cols_dif:

            dict_mapp = dict()

            for col_name_new in cols_dif:

                col_name_orig = col_name_new[0:col_name_new.rfind('_')]

                try:
                    dict_mapp[col_name_orig] = dict_mapp[col_name_orig] + \
                        [col_name_new]

                except Exception:

                    dict_mapp[col_name_orig] = [col_name_new]

            return dict_mapp

        return None

    # Inner function to get col list from cat type
    def _get_list(self, enc_type, cols_orig=False):
        """
        Get Encoded categorical colums list based on a cat type
        """
        try:
            if type(enc_type) == dict:

                columns = []
                keys_orig = []

                for key, values in enc_type.items():

                    columns = columns + values
                    keys_orig.append(key)

            else:
                columns = enc_type
                keys_orig = enc_type

            if cols_orig:
                columns = keys_orig

            return columns

        except Exception:

            return None

    # Inner function to get encoded cat variables into lists
    def _get_enc_categoricals(self, cols_orig=False):
        """
         Get Encoded categoricals eventually handling ohencoding specifity,
         returning a tuple of 3 arguments for the 3 types of categoricals
        """

        try:
            self._cols_ce_cathigh
        except Exception:
            self._cols_ce_cathigh = []

        try:
            self._cols_ce_catlow
        except Exception:
            self._cols_ce_catlow = []

        try:
            self._cols_ce_catord
        except Exception:
            self._cols_ce_catord = []

        enc_cathigh = self._get_list(self._cols_ce_cathigh,
                                     cols_orig=cols_orig)

        enc_catlow = self._get_list(self._cols_ce_catlow,
                                    cols_orig=cols_orig)

        enc_catord = self._get_list(self._cols_ce_catord,
                                    cols_orig=cols_orig)

        return (enc_cathigh, enc_catlow, enc_catord)

    # Inner function - Get encoded cat variables into lists
    def _get_list_cols(self, col_lists):
        """
         Get columns from provided lists, eventually handling
         ohencoding specifity, returning a list
        """

        numericals = []
        categoricals = []
        categoricals_orig = []

        for col_list in col_lists:

            if col_list in ['fmtcategory', 'fmtordcategory']:

                cat_lists = self._get_enc_categoricals()
                cat_lists_orig = self._get_enc_categoricals(cols_orig=True)

                for catorig, cat in zip(cat_lists_orig, cat_lists):

                    categoricals = categoricals + cat
                    categoricals_orig = categoricals_orig + catorig

                cat_notenc = [x for x in self.dict_cols[col_list]
                              if x not in categoricals_orig]

                categoricals = list(dict.fromkeys(
                        categoricals + cat_notenc).keys())

            else:

                numericals = numericals + self.dict_cols[col_list]

        return categoricals + numericals

    # Inner function for applying col, list or all mode
    def _apply_columns(self, col_lists, mode='all'):
        """
         Return the list of df columns in accordance to the
         given mode
        """
        if mode == 'all':
            columns = self.df.columns

        elif mode == 'fmt_names':
            columns = self._get_list_cols(col_lists)

        elif mode == 'col_names':
            columns = col_lists

        return columns

    # Inner function - Reset Set Test to Decoded Values
    def _reset_dftest(self):
        """
         Return the decoded test dataframe. Useful for analysis and
         explanation.
        """
        columns_init = [col for col in self.dftest.columns if
                        col in self._df_init_]

        columns_test = [col for col in self.dftest.columns if
                        col not in self._df_init_]

        df_tmp_init = self._df_init_.loc[self.dftest.index,
                                         columns_init]

        df_tmp_test = self.dftest[columns_test]

        df_decoded = df_tmp_test.merge(df_tmp_init,
                                       left_index=True, right_index=True)

        return df_decoded.sort_index().copy()

    # Split into 2 obj attributes categorical variables
    def set_categoricals(self, cardinality_split,
                         setslot=None):
        """
         Set two object attributes, each of one containing the list of the df
         categorical column names, belonging to the same cluster of variables.
         These two clusters of vars are genereted according to a cardinality
         categorical threshold, which is the function parameter.

         The two istance attributes are basis for other obj methods:
             set_encode methods, many of the EDA methods.

         Parameters
         _ _ _ _ _
         cardinality_split : integer
             The splitting threshold
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Two Lists
             self instance setting : fmtcat_high, fmtcat_low
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.set_categoricals,
                                    dict_locals,
                                    setslot)
            return

        fmtcat_high = []
        fmtcat_low = []

        for col in self.df[self.fmtcategory].columns:

            if len(self.df[col].unique()) > cardinality_split:

                fmtcat_high = fmtcat_high + [col]

            else:

                fmtcat_low = fmtcat_low + [col]

        self.fmtcat_high = fmtcat_high
        self.fmtcat_low = fmtcat_low

        # Outcome Message
        print('\n')
        print('The Cardinality split produced: \n')
        print(str(len(self.fmtcat_high)) + ' high cardinality columns')
        print(str(len(self.fmtcat_low)) + ' low cardinality columns')

    # Keep only columns and list passed to function
    def columns_keep(self, col_lists, mode='fmt_names', returns=False,
                     setslot=None):
        """
         In the istance dataframe keep only the colums belonged to the pandas
         data types given to the col_lists parameter.
         It also applies to test df if it exists.
         Accepted col_lists values : 'fmtcategory', 'fmtordcategory', 'fmtint',
         'fmtfloat', 'fmtdatetime'.

         Parameters
         _ _ _ _ _
         col_lists : list
             List of pandas data type labels or single columns
         mode : 'string'
             'all' : all of the dataframe columns
             'col_names' : if single columns are given
             'fmt_names : if pandas data types group labels are given
         returns : Boolean
             If to return the df to the calling python session
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         DataFrame
             self instance setting : df
             calling python session : df (optional)
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.columns_keep,
                                    dict_locals,
                                    setslot)
            return

        print('\n')

        # Drop not used list from dict_cols
        columns2keep = []

        if mode == 'fmt_names':

            for key in col_lists:

                columns2keep = columns2keep + self.dict_cols[key]

        else:
            
            columns2keep = col_lists[:]

        columnstot = []

        for key in self.dict_cols:

            columnstot = columnstot + self.dict_cols[key]

        columns2drop = [x for x in columnstot if x not in columns2keep]

        for col_name in columns2drop:

            self._drop_col(col_name, drop_df=False)

        # Keep Selected columns in df
        col_list = []
        for key in self.dict_cols:

            col_list = col_list + self.dict_cols[key]

        col_list = [x for x in self.df.columns if x in col_list]

        for key, df in self.__dict__.items():

            if type(df) == pd.DataFrame:

                df = df[col_list]

                df = df[np.sort(df.columns)]
                setattr(self, key, df)

                # Outcome Message
                print(key + ' has been updated')

                self.df.sort_index(inplace=True)

        if returns:
            return self.df

    # Drop columns from df, lists and dictionary
    def columns_drop(self, columns, drop_df=True,
                     setslot=None):
        """
         Drop provided df columns from the Fastlane

        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.columns_drop,
                                    dict_locals,
                                    setslot)
            return

        print('\n')
        for col in columns:

            self._drop_col(col, drop_df=True)

    # Drop columns completly nan
    def columns_dropnan(self, drop_df=True,
                        setslot=None):
        """
         Drop df columns having all of the values to missing (np.nan)

        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.columns_dropnan,
                                    dict_locals,
                                    setslot)
            return

        for col in self.df.columns:

            # If Column is completly nan
            if self.df[col].isna().mean() == 1:

                self.columns_drop(col)

    def columns_drop_correlated(self, absolute=None, threshold=0.8,
                                start_selector='model',
                                setslot=None):

        """
         Enable the user to drop the columns that have a
         correlation index higher than the given threshold.
         The index beneath the method is Pearson, whose range is from -1 to 1.
         It is possible to drop only the positively correlated ones, default,
         or to work on both sides, absolute=True.

         Parameters
         _ _ _ _ _
         absolute : Boolean
             If to delete correlated feature with no reference to the
             correlation sign
         threshold: float
             The correlation index threshold above which the cell is
             detected to be evaluated for deletion
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         ...
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.columns_drop_correlated,
                                    dict_locals,
                                    setslot)
            return

        if start_selector == 'model':

            feature_rank = self._model_select_stats.loc[
                    self._model_selkfold].sort_values(
                    'Mean_Kfold', ascending=False)[['Mean_Kfold']]

        elif start_selector == 'boruta':

            feature_rank = self._boruta_select_stats.loc[
                    self._boruta_selkfold]

            feature_rank = pd.DataFrame(feature_rank.iloc[:, -1].sort_values(
                    ascending=False))

        matrix = self.df.corr()
        cols2del = []

        print('\n')

        for ycol in matrix.columns:

            if absolute:

                y2delete = matrix.loc[matrix[ycol].apply(
                        lambda x: abs(x)) > threshold, ycol]

            else:

                y2delete = matrix.loc[matrix[ycol] > threshold, ycol]

            for pos, row in enumerate(y2delete):

                if y2delete.name != y2delete.index[pos]:

                    coly = y2delete.name
                    colx = y2delete.index[pos]

                    coly_value = feature_rank.loc[coly].values[0]
                    colx_value = feature_rank.loc[colx].values[0]

                    col2del = coly
                    col2keep = colx

                    if coly_value > colx_value:

                        col2del = colx
                        col2keep = coly

                    if col2del not in cols2del:

                        cols2del.append(col2del)

                        # Outcome Message
                        print('Comparing Feature Rank between ' +
                              str(col2del) + ' and ' + str(col2keep))

                        print(str(col2del) +
                              ' : showed a lower importance ranking\n')

        for col in cols2del:

            self._drop_col(col, drop_df=True)

    # Encoding high cardinality cat vars
    def encode_cathigh_set(self, encoder, enc_type='standard',
                           n_splits=5,
                           apply2test=False,
                           leave1out=True,
                           setslot=None):
        """
         Encode the categorical columns belonging to the high cardinality group

         Parameters
         _ _ _ _ _
         encoder : Category Encoder istance
             Expected an istance of one of the classes provided by the
             category_encoders Library.
         enc_type : String
             Two possible values : 'standard', 'kfold'
         n_splits : Integer
             The number of kfold the encoder estimator will be evaluated on
         apply2test : Boolean
             If to apply category encoder transform method on the test set
        leave1out: boolean
            Category encoders do not natively handle the option of leaving
            out from final dataframe one encoded level.
            This parameter gives such chance to the user.
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Encoded DataFrame
             self instance setting : df
         List of encoded column names:
             self instance setting : _cols_ce_cathigh
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.encode_cathigh_set,
                                    dict_locals,
                                    setslot)
            return

        columns = self.fmtcat_high
        cols_orig = self.df.columns

        self._set_encode(encoder, columns, enc_type, n_splits, apply2test)

        ohe_dict = self._get_onehotenc_dictcols(cols_orig)

        if ohe_dict:

            if leave1out:

                for key, value in ohe_dict.items():

                    col2del = value[-1]
                    ohe_dict[key].remove(col2del)

                    self.df.drop(columns=col2del, inplace=True)

                    if apply2test:
                        self.dftest.drop(columns=col2del, inplace=True)

            self._cols_ce_cathigh = ohe_dict

        else:
            self._cols_ce_cathigh = columns

        # Outcome Message
        print('\n')
        print('High Cardinality Categorical Variables')
        print('\n')
        print(str(len(self.fmtcat_high)) + ' columns have been encoded')

    # Encoding low cardinality cat vars
    def encode_catlow_set(self, encoder, enc_type='standard',
                          n_splits=5,
                          apply2test=False,
                          leave1out=True,
                          setslot=None):
        """
         Encode the categorical columns belonging to the low cardianlity group

         Notes
         _ _ _ _ _
         The same applied logic as encode_cathigh_set method
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.encode_catlow_set,
                                    dict_locals,
                                    setslot)
            return

        columns = self.fmtcat_low
        cols_orig = self.df.columns

        self._set_encode(encoder, columns, enc_type, n_splits, apply2test)

        ohe_dict = self._get_onehotenc_dictcols(cols_orig)

        if ohe_dict:

            if leave1out:

                for key, value in ohe_dict.items():

                    col2del = value[-1]
                    ohe_dict[key].remove(col2del)

                    self.df.drop(columns=col2del, inplace=True)

                    if apply2test:
                        self.dftest.drop(columns=col2del, inplace=True)

            self._cols_ce_catlow = ohe_dict

        else:
            self._cols_ce_catlow = columns

        # Outcome Message
        print('\n')
        print('Low Cardinality Categorical Variables')
        print('\n')
        print(str(len(self.fmtcat_low)) + ' columns have been encoded')

    # Encoding ordinal vars
    def encode_catord_set(self, encoder, enc_type='standard',
                          n_splits=5,
                          apply2test=False,
                          leave1out=True,
                          setslot=None):
        """
         Encode the categorical columns belonging to the cat ordinal group

         Notes
         _ _ _ _ _
         The same applied logic as encode_cathigh_set method
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.encode_catord_set,
                                    dict_locals,
                                    setslot)
            return

        columns = self.fmtordcategory
        cols_orig = self.df.columns

        self._set_encode(encoder, columns, enc_type, n_splits, apply2test)

        ohe_dict = self._get_onehotenc_dictcols(cols_orig)

        if ohe_dict:

            if leave1out:

                for key, value in ohe_dict.items():

                    col2del = value[-1]
                    ohe_dict[key].remove(col2del)

                    self.df.drop(columns=col2del, inplace=True)

                    if apply2test:
                        self.dftest.drop(columns=col2del, inplace=True)

            self._cols_ce_catord = ohe_dict

        else:
            self._cols_ce_catord = columns

        # Outcome Message
        print('\n')
        print('Ordinal Categorical Variables')
        print('\n')
        print(str(len(self.fmtordcategory)) + ' columns have been encoded')

    # Decoding categorical vars
    def decode_get(self, cat_type='cathigh', mode='replace',  returns=False,
                   apply2test=False,
                   setslot=None):
        """
         The reverse method of the set_encode ones. By applying the method
         is returned the decoded dataframe, in accordance to the choosen
         cat_type. In the update mode the encoded columns, along with the
         original ones (enriched with a _dec suffix),  will be kept in the
         istance dataframe.
         Note:
         Mode='update  is a valid option only for returning result
         to the python session, returns=True, nothing will be done to the
         internal object dataframes.
         The decode_get method does not work for dataframe which have
         been balanced by using generation techinques, such as ADASYIN,
         SMOOTHE, whose specificity is to create new table keys, threfore
         not compatible with the original records and keys.

         Parameters
         _ _ _ _ _
         cat_type : String
             'cathigh' : high cardinality group
             'catlow' : low cardinality group
             'catord': ordinal cat group
         mode : String
             'replace' : Replace the original colums to the encoded ones
             'update' : Keep the two types at the same time
         n_splits : Integer
             The number of kfold the encoder estimator will be evaluated on
         returns : Boolean
             If to return the df to the calling python session
         apply2test : Boolean
             If to also get decoded df test
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.


         Returns
         _ _ _ _ _
         DataFrame
             self instance setting : df - dftest (optional)
             calling session : df - dftest (optional)
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.decode_get,
                                    dict_locals,
                                    setslot)
            return

        if cat_type == 'cathigh':
            col2decode = self._cols_ce_cathigh
            self._cols_ce_cathigh = []

        elif cat_type == 'catlow':
            col2decode = self._cols_ce_catlow
            self._cols_ce_catlow = []

        elif cat_type == 'catord':
            col2decode = self._cols_ce_catord
            self._cols_ce_catord = []

        if type(col2decode) == dict:

            colorig = []
            col2decode_ = []

            for key, values in col2decode.items():
                colorig = colorig + [key]
                col2decode_ = col2decode_ + values

            col2decode = col2decode_[:]

        else:
            colorig = col2decode

        if mode == 'replace':

            df = self.df.drop(columns=col2decode).join(
                        self._df_init_[colorig])

            self.df = df[np.sort(df.columns)].sort_index().copy()

            # Outcome Message
            print('\n')
            print(str(len(colorig)) +
                  ' columns have been decoded. The df dataframe replaced')

            if apply2test:
                dftest = self.dftest.drop(columns=col2decode).join(
                        self._df_init_[colorig])

                self.dftest = dftest[np.sort(
                        dftest.columns)].sort_index().copy()

                # Outcome Message
                print('\n')
                print(str(len(colorig)) +
                      ' columns have been decoded.',
                      'The dftest dataframe replaced')

            if returns:

                if apply2test:
                    return (self.df, self.dftest)

                return self.df

        elif mode == 'update':

            df = self.df.join(self._df_init_[colorig],
                              rsuffix="_dec")

            # Outcome Message
            print('\n')
            print(str(len(colorig)) +
                  ' columns have been decoded. The df dataframe updated')

            if apply2test:
                dftest = self.dftest.join(self._df_init_[colorig],
                                          rsuffix="_dec")

                # Outcome Message
                print('\n')
                print(str(len(colorig)) +
                      ' columns have been decoded.',
                      'The dftest dataframe updated')

                return (self.df, self.dftest)

            return self.df

    # Prepare int with nan and Set Missing Indicator
    def engineer_missing_set(self, replace_intnan=-1,
                             setslot=None):
        """
         The main goal of the method is, firstly, to detect the integer value
         used in place of the np.nan by the developer at the time of
         BinaryLane instantiation, with reference to the integer data type
         columns and, secondly, to replace it with the new introduced NA pandas
         data type. Indeed, up to now, pandas do not allow the developer to
         give nan values to integer data type columns. Circumnavigating the
         block, the method will enable the missing imputer method to also
         work on those null cases.
         It is a pre-requisite, otherwise this kind of missingness would not
         be taken into account during the imputation.
         Furthermore it instantiate the sklearn MissingIndicator object.

         Parameters
         _ _ _ _ _
         replace_intnan : integer
             In case of integer df columns whose missing values have been
             imputed with an integer value, instead of np.nan, for format
             reason (np.nan is float, not accepted in int).
             By setting the parameter the integer value will be correctly
             interpreted as missing.
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.engineer_missing_set,
                                    dict_locals,
                                    setslot)
            return

        # Replace int value with nan
        self.df[self.fmtint] = self.df[
                self.fmtint].astype('Int32').replace(replace_intnan, np.nan)

        if hasattr(self, 'dftest'):

            self.dftest[self.fmtint] = self.dftest[
                    self.fmtint].astype('Int32').replace(replace_intnan,
                                                         np.nan)

            df = self.df.append(self.dftest)

        # Create the missing mask
        self.miss_mask = pd.DataFrame(columns=df.columns,
                                      index=df.index)

        self.miss_mask.iloc[:, :] = MissingIndicator(
                features='all').fit_transform(df)

        self.miss_mask.sort_index(inplace=True)

        # Outcome Message
        print('\n')
        print('The integer value : ' + str(replace_intnan) +
              ' has been set to nan')

    # Smooth Outlier
    def engineer_smooth_outlier(self, col_lists=['fmtfloat', 'fmtint'],
                                mode='fmt_names', multiplier=3,
                                apply2test=False, returns=False,
                                setslot=None):
        """
         Replace outlier values with an upper and a lower border value.
         The detection outlier threshold is given by the following formula:
             lower = mean - std * multiplier parameter
             upper = mean + std * multiplier parameter
         In the 'col_names' mode, single colums given to the parameter are
         used for smoothing. The column names must be provided in a list.

         Parameters
         _ _ _ _ _
         col_lists : list
             List of pandas data types labels or single columns
         mode : 'string'
             'all' : all of the dataframe columns
             'col_names' : if single columns are given
             'fmt_names : if pandas data types group labels are given
         multiplier : integer
             The multiplier to be applied to the detection formula
         apply2test : Boolean
             If to apply the transformation on the test set
         returns : Boolean
             If to return the df to the calling python session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         DataFrame
             self instance setting : smoothed df, df_test (if apply2test)
             calling python session :
                 smoothed df,  df_test (if apply2test) (optional)
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.engineer_smooth_outlier,
                                    dict_locals,
                                    setslot)
            return

        columns = self._apply_columns(col_lists, mode)
        counter = 0

        print('\n')
        for col in columns:

            mean = self.df[col].mean()
            std = self.df[col].std()
            cut_off = std * multiplier
            lower, upper = mean - cut_off, mean + cut_off

            nvalues = sum(self.df[col] > upper) + sum(self.df[col] < lower)

            self.df[col] = np.where(
                    self.df[col] > upper, upper,
                    self.df[col])

            self.df[col] = np.where(
                    self.df[col] < lower, lower,
                    self.df[col])

            # Outcome Message
            if nvalues != 0:

                counter = 1
                print(str(col) + ' : ' + str(nvalues) +
                      ' values have been smoothed.')

            if apply2test:

                self.dftest[col] = np.where(
                        self.dftest[col] > upper, upper,
                        self.dftest[col])

                self.dftest[col] = np.where(
                        self.dftest[col] < lower, lower,
                        self.dftest[col])

                nvalues = sum(self.dftest[col] > upper) + \
                    sum(self.dftest[col] < lower)

                if returns:
                    return (self.df[np.sort(self.df.columns)],
                            self.dftest[np.sort(self.dftest.columns)])

        # Outcome Message
        if counter == 0:

            print('None of the dataframe have been smoothed.')

        if returns:
            return self.df[np.sort(self.df.columns)]

    # Standardize df
    def engineer_standardize(self, col_lists=None, mode='all',
                             apply2test=False, returns=False,
                             setslot=None):
        """
         Standardize the dataframe colums by using the sklearn StandardScaler.
         The formula beneath the method is:
             mean - x / std
         Three mode are available for passing the columns to the method :
         With the first one 'all', default, the method will be applied
         on the entire dataframe.
         The second mode, 'fmt_names', enables to work with pandas data
         type group of columns. In such a case the col_lists param must be
         imputed with one or more of the followings (in list).

                 'fmtcategory' : categorical columns
                 'fmtordcategory' : ordinal categorical columns
                 'fmtint' : integer number columns
                 'fmtfloat' : float number columns
                 'fmtdatetime' : date and datetime columns

         In the 'col_names' mode, instead, the individual given colums
         (in list) are used for imputation.

         Parameters
         _ _ _ _ _
         col_lists : list
             List of pandas data type labels or single columns
         mode : 'string'
             'all' : all of the dataframe columns
             'col_names' : if single columns are given
             'fmt_names : if pandas data types group labels are given
         apply2test : Boolean
             If to apply the transformation on the test set
         returns : Boolean
             If to return the df to the calling session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         DataFrame
             self instance setting : standardized df, df_test (if apply2test)
             calling python session :
                 standardized df,  df_test (if apply2test) (optional)
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.engineer_standardize,
                                    dict_locals,
                                    setslot)
            return

        columns = self._apply_columns(col_lists, mode)

        scaler = StandardScaler()

        self.df[columns] = scaler.fit_transform(self.df[columns])

        if apply2test:

            self.dftest[columns] = scaler.transform(self.dftest[columns])

            if returns:
                return (self.df[np.sort(self.df.columns)],
                        self.dftest[np.sort(self.dftest.columns)])

        # Outcome Message
        print('\n')
        print(str(len(columns)) + ' columns have been standardized')

        if returns:
            return self.df

    # Simple Imputer
    def engineer_missing_imputer(self, imputer, col_lists=None, mode='all',
                                 apply2plot=None, apply2test=None,
                                 returns=False, setslot=None):
        """
         Missing imputer for imputing nan values with one of the available
         sklearn imputer objects, which must be given to the imputer param.
         The imputation is applied on the colums provided to the method,
         which can be done in three different modes.
         With the first one 'all', default, the method will be applied
         on the entire dataframe .
         The second mode, 'fmt_names', enables to work with pandas data
         type group of columns. In such a case the col_lists param must be
         imputed with one or more of the following.

                 'fmtcategory' : categorical columns
                 'fmtordcategory' : ordinal categorical columns
                 'fmtint' : integer number columns
                 'fmtfloat' : float number columns
                 'fmtdatetime' : date and datetime columns

         In the 'col_names' mode, instead, the individual given column
         (in list) are used for imputation.

         Parameters
         _ _ _ _ _
         imputer : sklearn imputer object
             All of the sklearn imputer of the imputer sub-module are accepted
         col_lists : list
             List of pandas data type labels or single columns
         mode : 'string'
             'all' : all of the dataframe columns
             'col_names' : if single columns are given
             'fmt_names : if pandas data types group labels are given
         apply2test : Boolean
             If to apply the transformation on the test set
         returns : Boolean
             If to return the df to the calling session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         DataFrame
             self instance setting : imputed df, df_test (if apply2test)
             calling python session :
                 imputed df,  df_test (if apply2test) (optional)
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.engineer_missing_imputer,
                                    dict_locals,
                                    setslot)
            return

        columns = self._apply_columns(col_lists, mode)

        if apply2plot:

            nan_columns = list(round(self.df.isna().mean()[
                    self.df.isna().mean() != 0], 2).index)

            self._df_preplot = self.df[nan_columns].copy()

        # Outcome Message
        print('\n')
        print('Columns with nan values at pre imputation time')
        print(round(self.df.isna().mean()[self.df.isna().mean() != 0], 2))

        self.df[columns] = imputer.fit_transform(self.df[columns])

        # Outcome Message
        print('\n')
        print('Columns with nan values after imputation')
        print(round(self.df.isna().mean()[self.df.isna().mean() != 0], 2))

        if apply2test:

            self.dftest[columns] = imputer.transform(self.dftest[columns])

            if returns:
                return (self.df[np.sort(self.df.columns)],
                        self.dftest[np.sort(self.dftest.columns)])

        if returns:
            return self.df

    # Pre and Post Imputation Plot Distributions
    def engineer_missing_plots(self, root, path,
                               figsize=(22, 14),
                               palette='deep', style='white',
                               left=0.07, right=0.95,
                               bottom=0.05, top=0.9,
                               wspace=0.18, hspace=0.22, fsize=12,
                               setslot=None):
        """
         In the same figure the feature distributions pre and post imputation
         are shown so that it is easly visible the imputer effect and
         eventually compare the impact of using one or another of the sklearn
         imputer objects on the dataframe columns.
         The plot is enabled from the apply2plot parameter of the
         engineer_missing_imputer method if set to True.
         If multiple imputers in the same session are applied on different
         df columns to each one of them an engineer_missing_plots method
         should be following.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette to display
         style : string
             Seaborn style to display
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         wspace : float
             Width space among the four subplots (horizontal)
         hspace : float
             High space among the four subplots (vertical)
         fsize : integer
             Font size of text plots
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figures plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.exploratory_numerical_plot,
                                    dict_locals,
                                    setslot)
            return

        def fpctfmt(x, pos):
            pctfmt = '{:,.2f}%'.format(x * 100.0)
            return pctfmt

        def fthousfmt(x, pos):
            thousfmt = '{:,d}'.format(int(x // 100))
            return thousfmt

        # pctfmt = FuncFormatter(fpctfmt)
        thousfmt = FuncFormatter(fthousfmt)

        for field in self._df_preplot.columns:

            options = self._options_plot_template(figsize=figsize,
                                                  palette=palette, style=style,
                                                  left=left, right=right,
                                                  bottom=bottom, top=top,
                                                  wspace=wspace, hspace=hspace,
                                                  fsize=fsize,
                                                  row=1, col=2)

            fig, gs, set_suptitle, set_plttitle, set_subtitle, \
                set_main, set_sub, cpalette, divpalette = options
            c_one = cpalette[0]

            fig.suptitle('Imputational Plots : ' + field,
                         weight=1000, fontstyle='oblique',
                         size=20, stretch=1000, color=c_one)

            # Plotting Grid Space

            ax0 = fig.add_subplot(gs[0, 0])
            ax0.set_title('Pre Imputation Feature Distribution', set_plttitle)

            ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
            ax1.set_title('Post Imputation Feature Distribution', set_plttitle)

            # Axis ax0 - Pre Imputation
            sns.distplot(self._df_preplot[field].dropna(),
                         ax=ax0)

            if self.df[field].max() > 1000:
                ax0.set_xlabel(xlabel='values in k', style='oblique')
                ax0.xaxis.set_major_formatter(thousfmt)

            else:
                ax0.set_xlabel(xlabel='')

            # Axis ax1 - Post Imputation
            sns.distplot(self.df[field],
                         ax=ax1)

            if self.df[field].max() > 1000:
                ax1.set_xlabel(xlabel='values in k', style='oblique')
                ax1.xaxis.set_major_formatter(thousfmt)

            else:
                ax1.set_xlabel(xlabel='')

            plt.savefig(root + path + '\\Missing_' + field +
                        '_plot.jpg', dpi=300)

            plt.close()

            # Outcome Message
            print('\n')
            print('Matplotlib Figure ' + field + '.jpg' +
                  ' exported to ' + root + path)

    # Method to reset the applied engineering trasformations
    def engineer_reset(self, keep_imputation=False,
                       apply2test=True, returns=False,
                       setslot=None):
        """
         Reset all of the transformed applied methods to the df, such as:
         categorical encoding, feature engineering, etc, keeping valid
         the previous drop columns and balancing operetions. In other
         words the method will not bring any changes to the dataframe
         shape. Please note that the keep_imputation=True need the
         engineer_missing_set has been run before.
         Warning: keep_imputation is beeing build
         Note:
         1) The engineer_reset method does not work for dataframe which
         have been balanced by using generation techinques, such as ADASYIN,
         SMOOTHE, whose specificity is to create new table keys, threfore
         not compatible with the original records and keys table that in turn
         represent the main goal of this method.
         2) The imputed values are kept the same as those left just before
         using this transformation, which might make them useless in cases
         which the column domains of the original values and the eventually
         transformeted ones are not comparable for.
         Example : an imputed value of a target encoded cat variable
         once retrieved the original category feature domain.
         3) Apply2test is applied only for returning values to python
         session. Indeed, in case a df test is found, it is reset the same
         as it is done for df, so that every object component is aligned.

         Parameters
         _ _ _ _ _
         keep_imputation : Boolean
             If to keep the imputed values on the decoded dataframe.
             In this way all but the imputed operation are dropped off
         apply2test : Boolean
             If to apply the transformation on the test set
         returns : Boolean
             If to return the df to the calling session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.engineer_reset,
                                    dict_locals,
                                    setslot)
            return

        columns = []
        for key, values in self.dict_cols.items():
            columns = columns + values

            try:
                self.decode_get(cat_type='cathigh', mode='replace')
            except Exception:
                pass
            try:
                self.decode_get(cat_type='catlow', mode='replace')
            except Exception:
                pass
            try:
                self.decode_get(cat_type='catord', mode='replace')
            except Exception:
                pass

#        if keep_imputation:
#
#            if hasattr(self, 'dftest'):
#                df = self.df.append(self.dftest)
#
#            cols_toimpute = [x for x in columns if x in self.miss_mask.columns]
#
#            df = df[cols_toimpute].reset_index().drop_duplicates(
#                    subset=self.key, inplace=True)
#
#            _df_init_ = self._df_init_.loc[
#                    df.index].sort_index().copy()
#
#            self._df_init_[cols_toimpute] = self._df_init_[cols_toimpute].mask(
#                    self.miss_mask[cols_toimpute], df)

        # Training
        _df_init_ = self._df_init_.loc[
                self.df.index].sort_index().copy()

        self.df = _df_init_[np.sort(columns)].sort_index().copy()

        self.yvals = self.yvals.sort_index().copy()

        # Outcome Message
        print('\n')
        print('df dataframe has been reset')

        # Test
        if hasattr(self, 'dftest'):

            _df_init_test_ = self._df_init_.loc[
                self.dftest.index].sort_index().copy()

            self.dftest = _df_init_test_[
                    np.sort(columns)].sort_index().copy()

            self.ytest = self.ytest.sort_index().copy()

            # Outcome Message
            print('dftest dataframe has been reset')

        if returns:

            if apply2test:

                return (self.df, self.dftest)

            return self.df

# =============================================================================
# # <----- ----->
# # DataFrame Handling : Balance & Split
# # <----- ----->
# =============================================================================

    # Train Test Split in a stratify fashion
    def train_test_split(self, test_size=0.2, random_state=0, returns=False,
                         setslot=None):
        """
         Train - Test split, based on train_test_split sklearn method.

         Parameters
         _ _ _ _ _
         test_size : float
         random_state : integer
         returns : Boolean
             If to return the df to the calling session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Train df, train y, Test df, test y
             self instance setting
             calling session (optional)
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.train_test_split,
                                    dict_locals,
                                    setslot)
            return

        self.df, self.dftest, self.yvals, self.ytest = train_test_split(
                self.df,
                self.yvals,
                stratify=self.yvals,
                test_size=test_size,
                random_state=random_state)

        self.df.sort_index(inplace=True)
        self.dftest.sort_index(inplace=True)
        self.yvals.sort_index(inplace=True)
        self.ytest.sort_index(inplace=True)

        # Outcome Message
        print('\n')
        print("Train Set : \n" + str(self.yvals.value_counts()) + "\n")
        print("Test Set : \n" + str(self.ytest.value_counts()))

        if returns:

            return (self.df, self.dftest, self.yvals, self.ytest)

    # Balance DataSet
    def df_balance(self, sampler, keep_prebalance=True, returns=False,
                   setslot=None):
        """
         Balancing dataframe based on imblearn library istances.
         Please note that the balance is always applied to only the training
         set if train_test_split method has already been run.

         Parameters
         _ _ _ _ _
         sampler : imblearn object class
             Select one of the available balancing classes available in the
             imblearn module.
             link : https://imbalanced-learn.readthedocs.io/en/stable/api.html
             Example : imblearn.under_sampling.NearMiss()
         keep_prebalance : Boolean
             if to store the original dataframe in an internal object
             attribute : _df_pre_balance.
             Useful if restoring the original df can be a possible developer
             need while developing the pipeline.
         returns : Boolean
             If to return the df to the calling python session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Balance df, y
             self instance setting
             calling python session (optional)
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.df_balance,
                                    dict_locals,
                                    setslot)
            return

        if keep_prebalance:
            self._df_pre_balance = self.df.copy()
            self._y_pre_balance = self.yvals.copy()

        # Outcome Message
        print('\n')
        print('Pre Balance Set : \n' + str(self.yvals.value_counts()) + '\n')

        X_resampled, y_resampled = sampler.fit_resample(self.df.reset_index(),
                                                        self.yvals)

        col_dtypes = self.df.dtypes.to_dict()

        df = pd.DataFrame(X_resampled,
                          columns=self.df.reset_index().columns)

        self.df = df.astype(col_dtypes)

        self.df.set_index(keys=self.key,
                          inplace=True)

        self.yvals = pd.Series(data=y_resampled, index=self.df.index,
                               name=self.y)

        self.df.sort_index(inplace=True)
        self.yvals.sort_index(inplace=True)

        # Outcome Message
        print('Post Balance Set : \n' + str(self.yvals.value_counts()) + '\n')

        try:
            train_ratio = round(len(self.yvals) /
                                (len(self.yvals) + len(self.ytest)), 2)

            test_ratio = round(len(self.ytest) /
                               (len(self.yvals) + len(self.ytest)), 2)

            # Outcome Message
            print('Train / Test Ratio : \n')
            print(str(train_ratio) + ' / ' + str(test_ratio))

        except Exception:
            pass

        if returns:

            return (self.df[np.sort(self.df.columns)], self.yvals)

    # Restore Pre Balance Df
    def set_prebalance(self,
                       setslot=None):
        """
         Restore the pre balanced dataframe

         Returns
         _ _ _ _ _
         Pre Balance df, y
             self instance setting
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.set_prebalance,
                                    dict_locals,
                                    setslot)
            return

        self.df = self._df_pre_balance.copy()
        self.yvals = self._y_pre_balance.copy()

        self.df.sort_index(inplace=True)
        self.yvals.sort_index(inplace=True)

        # Outcome Message
        print('\n')
        print('Dataframe returns to pre Balance Values: \n' +
              str(self.yvals.value_counts()) + '\n')

    # Pre & Post Balance Scatter Plot
    def balance_plot(self, root, path,
                     figsize=(24, 12),
                     bottom=0.05, top=0.95,
                     left=0.05, right=0.95,
                     setslot=None):
        """
         The main goal of the method is to give the developer a graphical
         overview about the difference in the dataframe target classes
         distribution afeter applying the df_balance method, so that in
         a glance is possible to see how the imblearn classifier operated.
         Without any statistical presumption, the method create two scatter
         plot figures on the basis of two derived axes.
         The x axis, represented by the Linear Discriminant Analysis
         component, while the y axis from the first PCA Component.
         For its innate nature, Linear Discriminant Analysis is the best
         available option for plotting the decision boundary on a
         classification task, but in binary projects the classifier only
         expone one component, due to the mathematical formulation at the
         base of the algorithm. To circumnavigate the problem, the method
         uses the Principal Component Analysis for the second axis, which
         probably is not statistically robust, but that practically gives the
         developer an useful summary view of the consequences of the balance
         operation.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         figsize: tuple
            To set Matplotilib figure size
         bottom: float
            Bottom Figure Margin
         top: float
            Top Figure Margin
         left: float
            Left Figure Margin
         right: float
            Right Figure Margin
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """

        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.balance_plot,
                                    dict_locals,
                                    setslot)
            return

        # Pre Balance PCA- LDA
        pca = PCA(n_components=2, random_state=0)
        lda = LDA()

        pca.fit(self._df_pre_balance)

        pca_prebalance = pd.DataFrame(pca.transform(self._df_pre_balance),
                                      columns=['PCA1', 'PCA2'])

        lda_prebalance = pd.DataFrame(lda.fit_transform(self._df_pre_balance,
                                                        self._y_pre_balance),
                                      columns=['LDA'])

        df_prebalance = pd.concat([pca_prebalance,
                                   lda_prebalance,
                                   self._y_pre_balance.reset_index(drop=True)],
                                  axis=1)

        # Post Balance PCA- LDA
        pca = PCA(n_components=2, random_state=0)
        lda = LDA()

        pca.fit(self.df)

        pca_online = pd.DataFrame(pca.transform(self.df),
                                  columns=['PCA1', 'PCA2'])

        lda_online = pd.DataFrame(lda.fit_transform(self.df,
                                                    self.yvals),
                                  columns=['LDA'])

        df_online = pd.concat([pca_online,
                              lda_online,
                              self.yvals.reset_index(drop=True)],
                              axis=1)

        # Figure Pre Balance
        fig, ax = plt.subplots(figsize=figsize)

        fig.suptitle('Comparison Plot: Pre Balance DataFrame',
                     weight=800, fontstyle='normal', size=12,
                     stretch=1000, horizontalalignment='center')

        sns.scatterplot(x='LDA', y='PCA1', hue=self.y,
                        data=df_prebalance, alpha=0.5,
                        legend='brief', ax=ax)

        fig.subplots_adjust(bottom=bottom, top=top,
                            left=left, right=right)

        plt.savefig(root + path + '\\Pre_Balance_Plot' + '.jpg',
                    dpi=300)

        # Figure Post Balance
        fig, ax = plt.subplots(figsize=figsize)

        fig.suptitle('Comparison Plot: Online DataFrame',
                     weight=800, fontstyle='normal', size=12,
                     stretch=1000, horizontalalignment='center')

        sns.scatterplot(x='LDA', y='PCA1', hue=self.y,
                        data=df_online, alpha=0.5,
                        legend='brief', ax=ax)

        fig.subplots_adjust(bottom=bottom, top=top,
                            left=left, right=right)

        plt.savefig(root + path + '\\Post_Balance_Plot' + '.jpg',
                    dpi=300)

        # Outcome Message
        print('\n')
        print('Pre_Balance_Plot.jpg and Post_Balance_Plot.jpg',
              'have been exported to' + str(root) + str(path) + '\n')

# =============================================================================
# # <----- ----->
# # Feature Selector
# # <----- ----->
# =============================================================================

    # Inner Function - Generic Univariate Selection function
    def _univariate_selector(self, features, selector):
        """
         Inner funcion for Univariate Feature Selection,
         based on statistical testing.
         The function is called by the main functions:
             - selector_univariate_cat, for fs selection of categoricals
             - selector_univariate_num, for fs selection of numericals
         On the base of a sklearn univariate selector object it returns
         the list of selected features and a dataframe of statistics.

         Parameters
         _ _ _ _ _
         features : List
             The list of features to be evaluated for selection
         selector : sklearn selector object
             Accepts instantiated object from the following sklearn classes:
                 SelectFwe : Select the p-values corresponding to FW error
                 SelectKBest :Select according to the k highest scores.
                 SelectPercentile : Select on a percentile of highest scores
                 SelectFpr :  Select the pvalues based on a FPR test
             link :
             https://scikit-learn.org/stable/modules/feature_selection.html

         Returns
         _ _ _ _ _
         Tuple ( list, dataframe )
             [0] The list of selected features
             [1] A Pandas df containing 3 fields:
                 Columns : The feature name
                 Scores : The test statistics, value score
                 Pvalues : The test statistics, pvalue
        """
        df = self.df[features].copy()

        for col in self.df[features].columns:

            if type(col) != int or type(col) != float:

                encoder = ce.OrdinalEncoder(cols=[col])

                df = encoder.fit_transform(df, self.yvals)

        selector = selector.fit(df[features], self.yvals)

        select_stats = pd.DataFrame(zip(df[features].columns,
                                        np.round(selector.scores_, 4),
                                        np.round(selector.pvalues_, 4)),
                                    columns=['Columns', 'Scores', 'Pvalues'])

        selected = [df[features].columns[i]
                    for i in range(0, len(selector.get_support()))
                    if selector.get_support()[i]]

        return (selected, select_stats)

    # Inner Function - Cv kfold statistics and scores
    def _model_kfoldfstats(self, classifier, folder, df, yvals):
        """
         Compute the score and feature_importance of the given classifier,
         in a cv kfold fashion

         Parameters
         _ _ _ _ _
         classifier : sklearn machine learning model
             Accepts any sklearn model as long as the feature_importance_
             classifier attribute is available
         folder : sklearn cross validate folder
             Accepts instantiated object from the following sklearn classes:
                 StratifiedKFold (default): Stratified K-Folds cross-validator
                 Kfold : K-Folds cross-validator
                 GroupKFold : K-fold iterator variant, non-overlapping groups
                 RepeatedKFold : Repeated K-Fold cross validator
                 RepeatedStratifiedKFold : Repeated Stratified K-Fold cv
            link :
            https://scikit-learn.org/stable/modules/generated/
            sklearn.model_selection.StratifiedKFold
         df : pandas Dataframe
             The input table which the model statistics are computed from
         yvals : pandas Series
             The target variable

         Returns
         _ _ _ _ _
         Tuple
         [0] select_stats : Pandas Daframe with classifier.feature_importances_
         [1] df_scores : Pandas Daframe with classifier.scores
        """
        gen_kfold = folder.split(df, yvals)

        score = list()

        # Outcome Message
        print('\n')
        print('Training and Validation Scores over k fold iterations :')

        for idx, (train, test) in enumerate(gen_kfold):

            classifier.fit(df.iloc[train],
                           yvals[train])

            score_train = round(classifier.score(df.iloc[train],
                                                 yvals[train]), 2)

            score_kfold = round(classifier.score(df.iloc[test],
                                                 yvals[test]), 2)

            col_importance = 'Importance_' + str(idx)

            df_kfold = pd.DataFrame(
                        data=np.round(classifier.feature_importances_, 3),
                        index=df.columns,
                        columns=[col_importance])

            score_name = 'Score_' + str(idx)

            score = {score_name: [score_train, score_kfold]}

            print(score)

            df_score = pd.DataFrame(data=score,
                                    index=['Train', 'Validation'])

            if idx == 0:
                select_stats = df_kfold.copy()
                df_scores = df_score.copy()

            else:
                select_stats = select_stats.join(df_kfold)
                df_scores = df_scores.join(df_score)

        select_stats['Mean_Kfold'] = select_stats.mean(axis=1)

        df_scores['Mean_Kfold'] = df_scores.mean(axis=1)

        for col in select_stats.columns:

            select_stats[col] = select_stats[col].apply(
                    lambda x: (x - select_stats[col].mean()) /
                    select_stats[col].std())

        return (select_stats, df_scores)

    # Inner Method Model Selection
    def _rfselection(self, classifier,
                     feature_importance_rfs, score_rfs,
                     cols2train, col_name, col2keep,
                     n_splits=5,
                     shuffle=True,
                     random_state=0):
        """
         ...

         Parameters
         _ _ _ _ _

         classifier : sklearn machine learning model
             Accepts any sklearn model as long as the feature_importance_
             classifier attribute is available
         feature_importance_rfs: pandas DataFrame
            The df containing the feature importance over the n iterations.
         score_rfs: pandas DataFrame
            The df containing the scores over the n iterations.
         cols2train: list
            List of columns the classifier is fitted on
         col_name: string
            The column name of the iteration
         col2keep: list
            The list of columns that are part of the final dfs
         n_splits : Integer
             The number of kfold the encoder estimator will be evaluated on
             sklearn.model_selection Kfold class is used for splitting.
             Valid only if mode = 'cv'.
         shuffle: Boolean
            The cv folder strategy. It yes the data is shuffled before
            beeing split in k folds.
             Valid only if mode = 'cv'.
         random_state: int
            A random state for reproducibility of results


         Returns
         _ _ _ _ _
         DataFrame of feature importance and model scores
        """
        cv_results = self._model_kfoldfstats(
                            classifier,
                            folder=StratifiedKFold(n_splits=n_splits,
                                                   shuffle=True,
                                                   random_state=0),
                            df=self.df[cols2train],
                            yvals=self.yvals)

        feature_importance = cv_results[0][['Mean_Kfold']]

        feature_importance.rename({'Mean_Kfold': col_name},
                                  axis=1, inplace=True)

        df_scores = cv_results[1]

        score_vals = [
                np.round(df_scores.loc['Train', 'Mean_Kfold'], 3),
                np.round(df_scores.loc['Validation', 'Mean_Kfold'], 3)]

        score = pd.DataFrame(
                    data=score_vals,
                    index=['score_train', 'score_validation'],
                    columns=[col_name])

        feature_importance_rfs = pd.concat([feature_importance_rfs.copy(),
                                           feature_importance],
                                           axis=1,
                                           names=col2keep)

        score_rfs = pd.concat([score_rfs,
                              score],
                              axis=1,
                              names=col2keep)

        return feature_importance_rfs, score_rfs

    # Inner Method Model forward Selection
    def _rf_forward_evaluation(self, feature_importance_rfs, score_rfs,
                               cols2train, col2keep, cols2print,
                               modelfit_threshold):
        """
         ...

         Parameters
         _ _ _ _ _

         feature_importance_rfs: pandas DataFrame
            The df containing the feature importance over the n iterations.
         score_rfs: pandas DataFrame
            The df containing the scores over the n iterations.
         cols2train: list
            List of columns the classifier is fitted on
         cols2print: string
            The column name of the iteration
         col2keep: list
            The list of columns that are part of the final dfs
        modelfit_threshold float
            The threshold to accept the introduction of the feature

         Returns
         _ _ _ _ _
         DataFrame of feature importance and model scores along with
         column list for internal function handling
        """
        outcome = 0

        if score_rfs.iloc[1, -1] <= score_rfs.iloc[1, -2] + modelfit_threshold:

            outcome = 1

            feature_importance_rfs = feature_importance_rfs.iloc[:, :-1]
            score_rfs = score_rfs.iloc[:, :-1]
            cols2train = cols2train[:-1]
            col2keep = col2keep[:-1]

            # Outcome Message
            print('\n')
            print(str(cols2print) +
                  ' : Removed from the model upon introduction')
            print('\n')

        else:
            # Outcome Message
            print('\n')
            print(str(cols2print) +
                  ' : kept in the model upon introduction')
            print('\n')

        return outcome, feature_importance_rfs, score_rfs, cols2train, col2keep

    # Inner Method Model backward Selection
    def _rf_backward_evaluation(self, feature_importance_rfs, score_rfs,
                                cols2train, col2keep, col, col_name,
                                modelfit_threshold):
        """
         ...

         Parameters
         _ _ _ _ _

         feature_importance_rfs: pandas DataFrame
            The df containing the feature importance over the n iterations.
         score_rfs: pandas DataFrame
            The df containing the scores over the n iterations.
         cols2train: list
            List of columns the classifier is fitted on
         cols2print: string
            The column name of the iteration
         col2keep: list
            The list of columns that are part of the final dfs
         modelfit_threshold : float
            The threshold to accept the remotion of the feature

         Returns
         _ _ _ _ _
         DataFrame of feature importance and model scores along with
         column list for internal function handling
        """
        outcome = 1

        if score_rfs.iloc[1, -2] - modelfit_threshold > score_rfs.iloc[1, -1]:

            outcome = 0

            feature_importance_rfs = feature_importance_rfs.iloc[:, :-1]
            score_rfs = score_rfs.iloc[:, :-1]

            # Outcome Message
            print('\n')
            print(str(col) +
                  ' : kept in the model after evaluation')
            print('\n')

        else:

            cols2train.remove(col)

            # Outcome Message
            print('\n')
            print(str(col) +
                  ' : Removed from the model after evaluation')
            print('\n')

        return outcome, feature_importance_rfs, score_rfs, cols2train, col2keep

    # Univariate Selection for categorical variables
    def selector_univariate_cat(self, cat_type=None,
                                selector=SelectFwe(chi2, 0.05),
                                returns=False,
                                setslot=None):
        """
         Univariate Feature Selection for categorical features,
         based on statistical testing.

         Parameters
         _ _ _ _ _
         cat_type : String
             If to evaluate only sub groups of categoricals.
             The default value is None for taking in account all
             of categorical columns
             - 'cathigh' : high cardinality columns
             - 'catlow' : low cardinality columns
             - 'catord' : ordinal categorical columns
         selector : sklearn selector object
             Accepts instantiated object from the following sklearn classes:
                 SelectFwe : Select the p-values corresponding to FW error
                 SelectKBest :Select according to the k highest scores.
                 SelectPercentile : Select on a percentile of highest scores
                 SelectFpr :  Select the pvalues based on a FPR test
             default : SelectFwe(chi2, 0.05)
             link :
             https://scikit-learn.org/stable/modules/feature_selection.html
         returns : boolean
             If to return the tuple to the calling python session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Tuple ( list, dataframe )
             [0] The list of selected features
             [1] A Pandas df containing 3 fields:
                 Columns : The feature name
                 Scores : The test statistics, value score
                 Pvalues : The test statistics, pvalue
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.selector_univariate_cat,
                                    dict_locals,
                                    setslot)
            return

        if cat_type == 'cathigh':
            features = self.fmtcat_high

        elif cat_type == 'catlow':
            features = self.fmtcat_low

        elif cat_type == 'catord':
            features = self.fmtordcategory

        else:
            features = []
            try:
                features = self.category
            except Exception:
                pass
            try:
                features = features + self.fmtordcategory
            except Exception:
                pass

        selected, select_stats = self._univariate_selector(features, selector)

        self._univariate_selcat = selected.copy()

        # Outcome Message
        print('\n')
        print(str(len(selected)) + ' categorical features selected')

        if returns:
            return (selected, select_stats)

    # Univariate Selection for numerical variables
    def selector_univariate_num(self, num_type=None,
                                selector=SelectFwe(f_classif, 0.05),
                                returns=False,
                                setslot=None):
        """
         Univariate Feature Selection for numerical features,
         based on statistical testing.

         Parameters
         _ _ _ _ _
         num_type : String
             If to evaluate only sub groups of numericals.
             The default value is None for taking in account all
             of numericals columns
             - 'fmtint' : integer colums
             - 'fmtfloat' : float colums
         selector : sklearn selector object
             Accepts instantiated object from the following sklearn classes:
                 SelectFwe : Select the p-values corresponding to FW error
                 SelectKBest :Select according to the k highest scores.
                 SelectPercentile : Select on a percentile of highest scores
                 SelectFpr :  Select the pvalues based on a FPR test
             default : SelectFwe(f_classif, 0.05)
             link :
             https://scikit-learn.org/stable/modules/feature_selection.html
         returns : boolean
             If to return the tuple to the calling python session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Tuple ( list, dataframe )
             [0] The list of selected features
             [1] A Pandas df containing 3 fields:
                 Columns : The feature name
                 Scores : The test statistics, value score
                 Pvalues : The test statistics, pvalue
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.selector_univariate_num,
                                    dict_locals,
                                    setslot)
            return

        if num_type == 'fmtint':
            features = self.fmtint

        elif num_type == 'fmtfloat':
            features = self.fmtfloat

        else:
            features = []
            try:
                features = self.fmtint
            except Exception:
                pass
            try:
                features = features + self.fmtfloat
            except Exception:
                pass

        selected, select_stats = self._univariate_selector(features, selector)

        self._univariate_selnum = selected.copy()

        # Outcome Message
        print('\n')
        print(str(len(selected)) + ' numerical features selected')

        if returns:
            return (selected, select_stats)

    # Not Linear Feature Selection
    def selector_kfold_model(self, classifier, threshold=0,
                             folder=StratifiedKFold(n_splits=5,
                                                    shuffle=True,
                                                    random_state=0),
                             returns=False,
                             setslot=None):
        """
         Stratified Cross Validation Model Feature Selection,
         based on sklearn model feature importance.

         Parameters
         _ _ _ _ _
         classifier : sklearn machine learning model
             Accepts any sklearn model as long as the feature_importance_
             classifier attribute is available
         threshold : float
             The feature importance acceptance threshold with range from 0 to 1
         folder : sklearn cross validate folder
             Accepts instantiated object from the following sklearn classes:
                 StratifiedKFold (default): Stratified K-Folds cross-validator
                 Kfold : K-Folds cross-validator
                 GroupKFold : K-fold iterator variant, non-overlapping groups
                 RepeatedKFold : Repeated K-Fold cross validator
                 RepeatedStratifiedKFold : Repeated Stratified K-Fold cv
            link :
            https://scikit-learn.org/stable/modules/generated/
            sklearn.model_selection.StratifiedKFold
         returns : boolean
             If to return the tuple to the calling python session
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Tuple (list, dataframe, model score)
             [0] The list of selected features
             [1] A Pandas df containing feature importance statistics
             [2] The default sklearn Model score
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.selector_kfold_model,
                                    dict_locals,
                                    setslot)
            return

        select_stats, score = self._model_kfoldfstats(classifier,
                                                      folder,
                                                      self.df,
                                                      self.yvals)

        selected = list(select_stats[
                select_stats['Mean_Kfold'] > threshold].index)

        self._model_selkfold = selected.copy()
        self._model_select_stats = select_stats.copy()

        # Outcome Message
        print('\n')
        print(str(len(selected)) + ' dataframe features selected')

        if returns:
            return (selected, select_stats, score)

    # Boruta Feature Selection
    def selector_boruta(self, model=None, importance_measure='Shap',
                        classification=True, percentile=100, pvalue=0.05,
                        n_trials=10, random_state=0, sample=True,
                        returns=False,
                        setslot=None):
        """
         Boruta Feature Selection,
         based on sklearn model feature importance.

         Parameters
         _ _ _ _ _
        model: Model Object
            If no model specified then a base Random Forest will be returned
            otherwise the specifed model will be returned.

        importance_measure: String
            Which importance measure too use either Shap or Gini/Gain

        classification: Boolean
            if true then the problem is either a binary or multiclass problem
            otherwise if false then it is regression

        percentile: Int
            An integer ranging from 0-100 it changes the value of the max
            shadow importance values. Thus, lowering its value
            would make the algorithm more lenient.

        p_value: float
            A float used as a significance level again if the p-value is
            increased the algorithm will be more lenient making it smaller
            would make it more strict also by making the model more strict
            could impact runtime making it slower. As it will be less likley
            to reject and accept features.

        n_trials : int
            The number of iterations the model will be trying to do

        random_state: int
            A random state for reproducibility of results

        Sample: Boolean
            if true then the a rowise sample of the data will be used
            to calculate the feature importance values

        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Tuple (list, dataframe, model score)
             [0] The list of selected features
                 [0] Accepted, [1] Tentative
             [1] A Pandas df containing feature importance statistics
             [2] The Boruta Column Hits
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.selector_boruta,
                                    dict_locals,
                                    setslot)
            return

        selector = BorutaShap(model=model,
                              importance_measure=importance_measure,
                              classification=classification,
                              percentile=percentile, pvalue=pvalue)

        selector.fit(self.df, self.yvals,
                     n_trials=n_trials, random_state=random_state,
                     sample=sample)

        hits = pd.Series(selector.hits, index=self.df.columns)

        select_stats = selector.history_x.transpose()
        select_stats.columns = ['Importance_' + str(col) for col
                                in select_stats.columns]

        selected = [selector.accepted, selector.tentative]

        self._boruta_selkfold = selector.accepted + selector.tentative
        self._boruta_select_stats = select_stats.copy()

        # Outcome Message
        print('\n')
        print(str(len(selected)) + ' dataframe features selected')

        if returns:
            return (selected, select_stats, hits)

    # Feature Importance Plot
    def selector_feature_plot(self, root, path, selector='model',
                              figsize=(18, 10),
                              bottom=0.3, top=0.9, plot_threshold=0,
                              setslot=None):
        """
         Plot the selected features on the x axis and the importance ranking
         on the y axis, on a point-plot fashion, giving the chance
         to choose among one of the class feature selectors : model selector,
         boruta selector. The figure si adjustable by using top and bottom
         parameters as well as the figure size one.
         The plot_threshold param is to limit the displayed features
         by filtering those which do not get the set limit. Please, note that
         the threshold is applied on the normalized column, so take it into
         account when choosing the value.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         selector: string
            Accepts the following two string values
                'model': if to plot model selected features
                'boruta': if to plot boruta selected features
         figsize: tuple
            To set Matplotilib figure size
         bottom: float
            Bottom Figure Margin
         top: float
            Top Figure Margin
         plot_threshold: float
            The minimum value for accepting plotting features
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.selector_feature_plot,
                                    dict_locals,
                                    setslot)
            return

        if selector == 'model':
            feature_importance = self._model_select_stats

        elif selector == 'boruta':
            feature_importance = self._boruta_select_stats

        columns = []
        for col in feature_importance.columns:

            if col[0:11] == 'Importance_':
                columns.append(col)

        # Prepare Figure
        col_label = 'Columns'
        rank_label = 'Feat_Rank'
        root_name = 'Feature_importance_'

        feature_importance = feature_importance[columns]

        feature_importance.dropna(axis=1, inplace=True)

        feature_importance = feature_importance.mean(axis=1).reset_index()

        feature_importance.columns = [col_label, rank_label]

        feature_importance = feature_importance[
                feature_importance[rank_label] > plot_threshold].sort_values(
                rank_label, ascending=False)

        # Plot Figure
        fig, ax = plt.subplots(figsize=figsize)

        fig.suptitle('Feature Importance Plot : ' + selector + ' Selector',
                     weight=800, fontstyle='normal', size=12,
                     stretch=1000, horizontalalignment='center')

        plt.subplots_adjust(bottom=bottom, top=top)

        ax = sns.pointplot(x=col_label,
                           y=rank_label,
                           data=feature_importance,
                           markers='^', linestyles='-')

        ax.set_xticklabels(feature_importance[col_label],
                           rotation='vertical')

        plt.savefig(root + path + '\\' + root_name + selector + '.jpg',
                    dpi=300)

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + root_name + selector + '.jpg' +
              ' exported to ' + root + path)

    # Apply Feature Selection
    def selector_apply(self, selector='boruta',
                       setslot=None):
        """
         Apply the choosen selector by deleting all the not selected
         features from the object dataframes and lists.
         By doing so the Fastlane components will keep in
         concordance of one another.

         Parameters
         _ _ _ _ _
         selector: string
            Accepts the following two string values
                'model': if to apply model selector
                'boruta': if to apply boruta selector
                'univariate': if to apply univariate selector
                'recursive': if to apply the recursive model selector
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Drop Columns Method applyied to all object components
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.selector_apply,
                                    dict_locals,
                                    setslot)
            return

        columns = []

        if selector == 'boruta':
            columns = self._boruta_selkfold

        elif selector == 'model':
            columns = self._model_selkfold

        elif selector == 'recursive':
            columns = list(self._explained_rfs[0].iloc[:, -1].dropna().index)

        # Get the selector to be applied
        elif selector == 'univariate':

            if hasattr(self, '_univariate_selnum') and \
                    hasattr(self, '_univariate_selcat'):

                columns = self._univariate_selnum + self._univariate_selcat

            elif hasattr(self, '_univariate_selnum'):

                columns = self._univariate_selnum

            elif hasattr(self, '_univariate_selcat'):

                columns = self._univariate_selcat

        # Get the list of columns to be deleted
        columns_dict = []

        for key, lista in self.dict_cols.items():

            columns_dict = columns_dict + lista

        columns_tot = list(dict.fromkeys(columns_dict +
                                         list(self.df.columns)))

        columns_todel = [col for col in columns_tot if col not in columns]

        # Apply the deletion
        print('\n')
        for col in columns_todel:

            self._drop_col(col)

        # Outcome Message
        print('\n')
        print(str(len(self.df.columns)) + ' columns have been kept')

    # Explained Feature Selection
    def recursive_model_selector(self, classifier,
                                 start_slot=5, start_selector='model',
                                 backward_threshold=-1,
                                 modelfit_threshold=0,
                                 n_splits=5, shuffle=True, random_state=0,
                                 returns=False,
                                 setslot=None):
        """
         ...

     Parameters
     _ _ _ _ _
     classifier : sklearn machine learning model
         Accepts any sklearn model as long as the feature_importance_
         classifier attribute is available
     start_slot: integer
        The number of features the model will be using in the first
        recursive round, in other word the starting dataframe to evaluate.
        The slot cursor starts from the positions taken by the features
        which showed to have the lowest importance rank in the
        model selector.
     start_selector: String
         'model': if to start from model selector ranking
         'boruta': if to start from model selector ranking
     backward_threshold: integer
         The threshold below which the feature is evaluated for remotion
         during the recursive iteration steps.
         Please, note that the threshold is applied on the normalized
         column, so take it into account when choosing the value.
     modelfit_threshold: integer
         The minimum model score improvement for accepting the introduction
         of a new feature during the forward step or for removing it during
         the backward steps. The value to give must be coherent with the
         model score, such as an example if the model score is the
         accuracy, providing a value of 1 would mean that a new feature
         is accepted only if brings a model improvement of one
         accuracy percentage point, and an old feature would be removed
         only if the model would loose an accuracy value of less than one
         percentage point.
     n_splits : Integer
         The number of kfold the encoder estimator will be evaluated on
         sklearn.model_selection Kfold class is used for splitting.
         Valid only if mode = 'cv'.
     shuffle: Boolean
        The cv folder strategy. It yes the data is shuffled before
        beeing split in k folds.
         Valid only if mode = 'cv'.
     random_state: int
        A random state for reproducibility of results
     returns : boolean
         If to return the tuple to the calling python session
     setslot : integer
         Please, refer to the constructor method for a wider
         parameter description.


         Returns
         _ _ _ _ _
         The recursive feature importance and score dataframes
    """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.recursive_model_selector,
                                    dict_locals,
                                    setslot)
            return

        if start_selector == 'model':

            df2explain = self._model_select_stats.loc[
                    self._model_selkfold].sort_values(
                    'Mean_Kfold')[['Mean_Kfold']]

        elif start_selector == 'boruta':

            _df2explain = self._boruta_select_stats.loc[
                    self._boruta_selkfold]

            df2explain = pd.DataFrame(
                    _df2explain.iloc[:, -1].sort_values())

        totround = len(df2explain.index) - start_slot

        feature_importance_rfs = pd.DataFrame()

        score_rfs = pd.DataFrame()

        cols2train = list()

        for nround in tqdm(range(0, totround + 1)):

            # Forward Selection - Variable introduction
            if nround == 0:
                cols2train = list(df2explain.iloc[:start_slot, :].index)
                col_name = 'T0'
                col2keep = [col_name]
                cols2print = cols2train

            else:
                pos = start_slot + nround - 1
                newcol2train = df2explain.iloc[[pos]].index[0]
                col_name = 'N_' + newcol2train
                cols2train.append(newcol2train)
                col2keep.append(col_name)
                cols2print = newcol2train

            # Outcome Message
            print('\n')
            print(str(cols2print) +
                  ' : Intoduced in the model')

            feature_importance_rfs, score_rfs = self._rfselection(
                    classifier,
                    feature_importance_rfs, score_rfs,
                    cols2train, col_name, col2keep,
                    n_splits=n_splits,
                    shuffle=shuffle,
                    random_state=random_state)

            if nround != 0:

                # Forward Selection - Score Evaluation
                forward_removed, feature_importance_rfs, \
                    score_rfs, cols2train, col2keep = \
                    self._rf_forward_evaluation(
                            feature_importance_rfs, score_rfs,
                            cols2train, col2keep, cols2print,
                            modelfit_threshold)

                if not forward_removed:

                    # Backward Selection - Variable Remotion
                    df2remove = feature_importance_rfs[[col_name]].dropna()
                    cols2evaluate = df2remove.loc[
                            df2remove[col_name] <= backward_threshold,
                            col_name].sort_values(ascending=True)

                    if any(cols2evaluate):

                        backward_outcome = 1

                        for col in cols2evaluate.index:

                            if backward_outcome:

                                # Outcome Message
                                print('\n')
                                print(str(col) + ' : In evaluation for ' +
                                      'backward selection')

                                print('Feature Importance Value : ' +
                                      str(round(
                                              cols2evaluate.loc[col], 2)))

                                col_name = 'R_' + str(col)

                                feature_importance_rfs, score_rfs = \
                                    self._rfselection(
                                        classifier,
                                        feature_importance_rfs, score_rfs,
                                        cols2train, col_name, col2keep,
                                        n_splits=n_splits,
                                        shuffle=shuffle,
                                        random_state=random_state)

                                # Backward Selection - Score Evaluation
                                backward_outcome, feature_importance_rfs, \
                                    score_rfs, cols2train, col2keep = \
                                    self._rf_backward_evaluation(
                                            feature_importance_rfs,
                                            score_rfs,
                                            cols2train, col2keep,
                                            col, col_name,
                                            modelfit_threshold)

                print('\n')

        self._explained_rfs = (feature_importance_rfs, score_rfs)

        if returns:
            return (feature_importance_rfs, score_rfs)

    # Explained RFS Plot
    def recursive_selector_plot(self, root, path, filename, filtercols=1,
                                feature_threshold=None,
                                figsize=(22, 14),
                                palette='dark', style='white',
                                left=0.05, right=0.95,
                                bottom=0.3, top=0.9,
                                wspace=0.25, hspace=0.22,
                                xrotation='vertical',
                                legend_cols=9,
                                setslot=None):
        """
         Inner Function for setting the base template options for the
         EDA plot functions .

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         filename: String
            The filename the figure is exported to
         figsize : tuple
             To set Matplotilib figure size
         filtercols: list or int
             If list only the df columns contained in the list
             will be plotted. If integer, only the features which
             have been kept until the -integer step will be kept.
         feature_threshold: integer
             If valued only the features that in the last iteration
             holds a normalized feature importance rank greater than the
             threshold will be showed
         palette : string
             Seaborn palette name to display
         style : string
             Seaborn style name to display
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         wspace : float
             Width space among the four subplots (horizontal)
         hspace : float
             High space among the four subplots (vertical)
         xrotation : string
             Matplotilib parameter of set_xticklabels method for
             arranging thx x axis ticks rotation of the feature
             importance subplot.
             Allowed values are:
                 'vertical': default
                 'horizontal'
                 angle in degrees : float from 0 to 1
        legend_cols: integer
             Number of columns of the legend

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.recursive_selector_plot,
                                    dict_locals,
                                    setslot)
            return

        row = 1
        col = 1

        opts = self._options_plot_template(figsize=figsize,
                                           palette=palette,
                                           style=style,
                                           left=left,
                                           right=right,
                                           bottom=bottom,
                                           top=top,
                                           wspace=wspace,
                                           hspace=hspace,
                                           row=row, col=col)

        fig, gs, set_suptitle, set_plttitle, set_subtitle, \
            set_main, set_sub, cpalette, divpalette = opts

        iterstep = 'Iteretaion_Step'
        rank = 'Importance_Ranking'
        cols = 'Columns'

        feature_importance, score = self._explained_rfs

        if feature_threshold is not None:

            feature_importance = feature_importance[
                    feature_importance.iloc[:, -1] > feature_threshold]

        if type(filtercols) == list:
            feature_importance = feature_importance.loc[filtercols, :]

        feature_importance = feature_importance.reset_index()
        feature_importance = pd.melt(feature_importance,
                                     id_vars=['index'],
                                     var_name=iterstep,
                                     value_name=rank).dropna(
                                             ).rename(columns={'index': cols})

        if type(filtercols) == int:

            iter2plot_unique = feature_importance.drop_duplicates(
                    subset=iterstep)[iterstep][-filtercols:].to_list()

            cols2plot = list(feature_importance[
                    feature_importance[iterstep].isin(iter2plot_unique)][
                            cols].unique())

            feature_importance = feature_importance[
                    feature_importance[cols].isin(cols2plot)]

        title1 = 'Recursive Model Selection'
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title(title1, set_plttitle)

        ax1.set_ylabel('Feature Normalized Importance Ranking')
        sns.pointplot(x=iterstep, y=rank,
                      hue=cols,
                      data=feature_importance,
                      palette=palette, hue_order=None, order=None,
                      markers='o', linestyles='-',
                      ci=None, n_boot=None, scale=0.7,
                      ax=ax1)

        ax1.set_xticklabels(feature_importance['Iteretaion_Step'].unique(),
                            rotation=xrotation)

        plt.legend(bbox_to_anchor=(0, -0.4, 0, 0),
                   loc='lower left', ncol=legend_cols,
                   title='Feature Legend')

        score = score.loc[:, feature_importance[iterstep].unique()]

        def fpctfmt(x, pos):
            pctfmt = '{:,.0f}%'.format(x * 100.0)
            return pctfmt

        pctfmt = FuncFormatter(fpctfmt)

        ax2 = ax1.twinx()

        ax2.set_ylabel('Train and Validation Scores')

        ax2.plot(score.columns,
                 score.loc['score_train'].values,
                 marker="1", linestyle="--", color='brown',
                 label='Training Score')

        ax2.plot(score.columns,
                 score.loc['score_validation'].values,
                 marker="1", linestyle="--", color='k',
                 label='Validation Score')

        ax2.yaxis.set_major_formatter(pctfmt)
        ax2.set_ylim(bottom=0, top=1)

        plt.legend(loc='upper left')

        fig.tight_layout()

        plt.savefig(root + path + '\\' + filename + '.jpg',
                    dpi=300)

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + filename + '.jpg' +
              ' exported to ' + root + path)

# =============================================================================
# # <----- ----->
# # Machine Learning Models
# # <----- ----->
# =============================================================================

    # Inner Method for getting the model from container
    def _get_model(self, model_name):
        """
         Inner method to retrieve model statistics and score,
         by passing through the model name.

         Parameters
         _ _ _ _ _
         model_name : string
             The name given to the model

         Returns
         _ _ _ _ _
         Tuple
             dictionary of metrics
             classifier : sklearn object (or compatible)
             ypred : array of predictions
             yprob : array of probabilities (positive case)
             model_features : feature importance (if allowed)
             model_results : if gridsearchCV
             model_name : name given to the model
        """
        for dict_model in self.models_container:

            if model_name in dict_model:

                dictm = dict_model[model_name][1]
                classifier = dict_model[model_name][0]

                ypred = dict_model[model_name][1]['model_ypredict']
                yprob = dict_model[model_name][1]['model_yprob'][:, 1]

                model_features = None
                model_results = None

                if 'model_features' in dict_model[model_name][1]:

                    model_features = dict_model[model_name][1][
                            'model_features']

                if 'model_results' in dict_model[model_name][1]:

                    model_results = dict_model[model_name][1][
                            'model_results']

                return (dictm, classifier, ypred, yprob,
                        model_features, model_results, model_name)

            else:
                pass

        print("Could not find the model in Fastlane")
        raise KeyError

    # Inner Method for getting the model score
    def _get_score(self, score_type):
        """
         Inner method to retrieve the dictionary of the model
         which shows to return the best choosen score among trained
         models. To do so, it is part of the model_set_best method.
         The dicionary is that holded in the model container.

         Parameters
         _ _ _ _ _
         score_type : string
             The score which the models are avaluated on.
             Allowed values are:
                score_accuracy
                score_precision
                score_recall
                score_roc_auc
                score_f1
                score_neg_log_loss
                model_score

         Returns
         _ _ _ _ _
         Dicionary
             key : model name
             values : [classifier, dict_metrics]
                 classifier: the fitted classifier object
                 dict_metrics: dict with all the score
                 and model statistics
        """
        score = 0
        for dict_model in self.models_container:

            for key, value in dict_model.items():

                score_new = value[1][score_type]

                if score_new > score:

                    best_model = dict_model

                    score = score_new

        return best_model

    # Inner Method for confusion matrix Template
    def _confusion_matrix_axes(self, ax, model_name,
                               divpalette=sns.diverging_palette(
                                       240, 10, n=18)):
        """
         Inner Function for preparing the confusion matrix template.

         Parameters
         _ _ _ _ _
         ax : Matplotlib axes
             A matplotlib axes the method is plotted on
         model_name : String
             The name given to the model
         divpalette : seaborn palette
             All the seaborn diverging palette are accepted

         Returns
         _ _ _ _ _
         Matplotlib axes setting
        """
        ypred = self._get_model(model_name)[2]

        annotations = {"ha": 'center', "va": 'center', "size": 14}

        ax.set_title('Confusion Matrix',
                     weight=800, fontstyle='normal', size=12,
                     stretch=1000, horizontalalignment='center')

        matrix = confusion_matrix(self.ytest, ypred)

        sns.heatmap(matrix, square=True, annot=True, annot_kws=annotations,
                    cbar=False, linewidths=.5, fmt=",d",
                    cmap=divpalette, ax=ax)

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('True - Ground Truth')

        return ax

    # Inner Method Prepare data for ROC Curves Plot
    def _roc_curve_data(self, model_name, rate=0.1):
        """
         Inner Method to prepare data for plotting the model
         ROC Curve.

         Parameters
         _ _ _ _ _
         model_name : String
             The name given to the model
         rate : float
             The rate of the thresholds which the tpr and fpr are
             evaluated on. Greater values means a more stiff curve,
             a lower floating point, instead, will generate a more
             detailed and gentle curve.

         Returns
         _ _ _ _ _
         Pandas DataFrame
             df Columns: 'Classifier', x', 'TrPsRate', 'FsPsRate', 'accuracy'
        """
        yprob = self._get_model(model_name)[3]

        ROC_dataframe = pd.DataFrame()

        yprob = pd.Series(yprob)
        ytest = self.ytest.reset_index(drop=True).copy()

        # Looping for thresholds
        for i in np.arange(0, 1.1, rate).round(1):

            y_class = yprob.where(yprob > i, other=0).where(yprob < i, other=1)

            accuracy = accuracy_score(ytest, y_class, normalize=True)

            matrix = confusion_matrix(ytest, y_class)

            mtx00 = matrix[0, :][0]
            mtx01 = matrix[0, :][1]
            mtx10 = matrix[1, :][0]
            mtx11 = matrix[1, :][1]

            # False Positive Rate
            if mtx01 == 0:
                fspr = 0

            else:
                fspr = (mtx01/(mtx01+mtx00)).round(2)

            # True Positive Rate
            if matrix[1, :][1] == 0:
                trpr = 0

            else:
                trpr = (mtx11/(mtx11+mtx10)).round(2)

            dictionary = {'Classifier': model_name,
                          'x': i,
                          'TrPsRate': trpr,
                          'FsPsRate': fspr,
                          'accuracy': accuracy}

            ROC_dataframe = ROC_dataframe.append(dictionary,
                                                 ignore_index=True)

        # Draw the Null model line
        for i in np.arange(0, 1.1, rate).round(1):

            dictionary = {'Classifier': 'Null Model',
                          'x': i,
                          'TrPsRate': i,
                          'FsPsRate': i,
                          'accuracy': 0}

            ROC_dataframe = ROC_dataframe.append(dictionary,
                                                 ignore_index=True)

        return ROC_dataframe

    # Inner Method for ROC Template
    def _roc_curve_axes(self, ax, ROC_dataframe, accuracy=True,
                        color_accuracy='brown'):
        """
         Inner Function for preparing the roc curve template.

         Parameters
         _ _ _ _ _
         ax : Matplotlib axes
             The matplotlib axes the method is plotted on
         ROC_dataframe : Pandas DataFrame
             A dataframe structured as that one outputted from
             the _roc_curve_data method
         accuracy : Boolean
             if to plot on the right y axes the accuracy percentage
             in correspondence of threshold floating points
         color_accuracy : Matplotlib color
             The accuray markers and y axes color to give
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib axes setting
        """
        def fpctfmt(x, pos):
            pctfmt = '{:,.0f}%'.format(x * 100.0)
            return pctfmt

        pctfmt = FuncFormatter(fpctfmt)
        ncolors = len(ROC_dataframe['Classifier'].unique())

        ax.set_title('ROC Curve',
                     weight=800, fontstyle='normal', size=12,
                     stretch=1000, horizontalalignment='center')

        sns.lineplot(x='FsPsRate',
                     y='TrPsRate',
                     hue='Classifier',
                     data=ROC_dataframe,
                     palette=sns.color_palette()[0:ncolors],
                     ax=ax)

        if accuracy:

            ax_2 = ax.twinx()
            ax_2.plot(ROC_dataframe.loc[
                    ROC_dataframe['Classifier'] != 'Null Model', 'FsPsRate'],
                      ROC_dataframe.loc[
                              ROC_dataframe['Classifier'] != 'Null Model',
                              'accuracy'],
                      marker='1', linestyle='None', color=color_accuracy)

            ax_2.set_ylim(bottom=0.04, top=1.04)
            ax_2.set_yticklabels([0.01, 0.2, 0.4, 0.6, 0.8, 1],
                                 color=color_accuracy)
            ax_2.set_ylabel('Accuracy Score', color=color_accuracy)
            ax_2.yaxis.set_major_formatter(pctfmt)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

    # Inner Method - Prepare data for Feature Importance Plot
    def _feature_importance_data(self, model_name, plot_threshold=0):
        """
         Inner Method to prepare data for plotting the model
         Feature Importance Ranking.
         Notice that the plot is only available  for those
         classifiers which provide the feature_importance_
         attribute.

         Parameters
         _ _ _ _ _
         model_name : String
             The name given to the model
         plot_threshold : float
             The plot_threshold param is to limit the features we
             want to display, by filtering those which do not get
             the limit set. Please, note that the threshold is applied
             on the normalized column, so take it into account
             when choosing the value.

         Returns
         _ _ _ _ _
         Tuple
             col_label: a string 'Columns'. X axes name.
             rank_label: a string 'Feat_Rank'. y axis name.
             feature_importance: dataframe with ft ranking
        """
        model_features = self._get_model(model_name)[4]

        col_label = 'Columns'
        rank_label = 'Feat_Rank'

        feature_importance = model_features.reset_index()

        feature_importance.columns = [col_label, rank_label]

        feature_importance = feature_importance[
            feature_importance[rank_label] > plot_threshold].sort_values(
            rank_label, ascending=False)

        return (col_label, rank_label, feature_importance)

    # Inner Method for Feature Importance Template
    def _feature_importance_axes(self, ax, feature_data,
                                 rotation='vertical'):
        """
         Inner Function for preparing the feature importance
         template.

         Parameters
         _ _ _ _ _
         ax : Matplotlib axes
             The matplotlib axes the method is plotted on
         feature_data : Tuple
             A tuple structured as that one outputted from
             the _feature_importance_data method
         rotation : String
             Matplotilib parameter of set_xticklabels.
             Allowed values are:
                 'vertical': default
                 'horizontal'
                 angle in degrees : float from 0 to 1

         Returns
         _ _ _ _ _
         Matplotlib axes setting
        """
        col_label, rank_label, feature_importance = feature_data

        ax.set_title('Feature Importance',
                     weight=800, fontstyle='normal', size=12,
                     stretch=1000, horizontalalignment='center')

        sns.pointplot(x=col_label,
                      y=rank_label,
                      data=feature_importance,
                      markers='^', linestyles='-', ax=ax)

        ax.set_xticklabels(feature_importance[col_label],
                           rotation='vertical')

        return ax

    # Inner Method to prepare grid search metrics
    def _grid_search_metrics(self, model_name):
        """
         Inner Method to prepare data for plotting the gridsearchCV
         statistics.

         Parameters
         _ _ _ _ _
         model_name : String
             The name given to the gridsearchCV model

         Returns
         _ _ _ _ _
         DataFrame
             DF Grouped for gridsearch choosen gridparams and
             summarized for the following two metrics:
             'mean_test_score': avarege score of hold out sets
             'mean_train_score': avarage score of training sets
        """
        model_results = self._get_model(model_name)[5]

        col_results = [col for col in list(model_results.columns)
                       if col[0:6] == 'param_']

        dic_pstat = dict()

        for col in col_results:

            col_name = col[6:]

            dic_pstat[col_name] = model_results.groupby(col)[
                    'mean_test_score', 'mean_train_score'].mean()

        return dic_pstat

    # Trein the Model and compute scores
    def model_fit_ingestion(self, model_name, classifier,
                            returns=False, keep_track=True,
                            setslot=None):
        """
         It is the enabler for developing machine learning models.
         The method getting a classifier object (mainly sklearn) and a model
         name, fit the model, computes many scores and statistics (below
         for details) and ingests them in an attribute container, which in
         turn will be the enabler for plot and explainable ai methods.
         Score and Computed Statistics:
             -accuracy score
             -auc score
             -log-loss score
             -f1, precision, recall score
             -mean test, prediction values
             -feature importance ranks if provided by classifier
             -train, validation scores if GridSearchCV
             -paramas statistics and best params if GridSearchCV

         Parameters
         _ _ _ _ _
         model_name : string
             The model name to give. Note: the name must be unique. Give
             the same name to different models might cause unexpected results.
             Take also into account that the given name will be required by
             some of the following plot methods and used for link and
             subtitles. therefore, choose a name suitable for the use.
         classifier : sklearn classifier
             Any of the sklearn model classifier objects are accepted as
             long as any other classifier which has a fit method, beeing
             sklearn compatible.
             The best solution for maximazing benefits of ready to use plots
             is to pass a GridSearchCV sklearn classifier which enable a
             greater amount of scores and statistics. Are accepted only
             GridSearchCV objects with Refit and train_score params
             set to True.
         returns : boolean
             If true return the model dictionary to the python session
         keep_track : boolean
             If true the model is be appended to the modelcontainer,
             otherwise it modelcontainer is reset and rubuild from scratch.
             By keeping track, the following plot methods will be produced
             for all the models embedded in the container in a loop fashion.
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         List of Dicionary
             self instance setting : model_container
                 container of all the trained models
                 (below for each element detail)
             calling python session : dicionary (optional)
             key : model name
             values : [classifier, dict_metrics]
                 classifier: the fitted classifier object
                 dict_metrics: dict with all the score
                 and model statistics
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.model_fit_ingestion,
                                    dict_locals,
                                    setslot)
            return

        def rounds(model_score):
            return np.around(model_score, decimals=2)

        classifier.fit(self.df, self.yvals)

        dict_model = dict()
        dict_metrics = dict()

        # Test Statistics
        dict_metrics['ytest_0'] = len(self.ytest[
                self.ytest == 0])

        dict_metrics['ytest_1'] = len(self.ytest[
                self.ytest == 1])

        dict_metrics['ytest_pmean'] = rounds(self.ytest.mean())

        # Model Methods
        dict_metrics['model_ypredict'] = classifier.predict(
                self.dftest)

        dict_metrics['model_yprob'] = classifier.predict_proba(
                self.dftest)

        dict_metrics['model_score_name'] = 'accuracy'

        dict_metrics['model_score'] = np.around(classifier.score(
                self.dftest, self.ytest), decimals=2)

        if hasattr(classifier, 'feature_importances_'):

            dict_metrics['model_features'] = pd.Series(
                    dict(zip(self.df.columns, np.around(
                            classifier.feature_importances_, 3))))

        # Model Scores
        ypred = dict_metrics['model_ypredict']
        yprob = dict_metrics['model_yprob']
        ytest = self.ytest.copy()

        dict_metrics['score_accuracy'] = rounds(metrics.accuracy_score(
                ytest, ypred))

        dict_metrics['score_precision'] = rounds(metrics.precision_score(
                ytest, ypred))

        dict_metrics['score_recall'] = rounds(metrics.recall_score(
                ytest, ypred))

        dict_metrics['score_roc_auc'] = rounds(metrics.roc_auc_score(
                ytest, ypred))

        dict_metrics['score_f1'] = rounds(metrics.f1_score(
                ytest, ypred))

        dict_metrics['score_neg_log_loss'] = rounds(metrics.log_loss(
                ytest, yprob))

        dict_metrics['model_pmean'] = rounds(yprob[:, 1].mean())

        # Grid Search Model
        if type(classifier) == GridSearchCV:

            dict_metrics['model_score_name'] = classifier.get_params(
                    )['scoring']

            dict_metrics['model_results'] = pd.DataFrame(
                    classifier.cv_results_)

            dict_metrics['model_bestparms'] = classifier.best_params_

            if hasattr(classifier.best_estimator_, 'feature_importances_'):

                dict_metrics['model_features'] = pd.Series(
                        dict(zip(self.df.columns, np.around(
                                classifier.best_estimator_.
                                feature_importances_, 3))))

            dict_metrics['train_score'] = rounds(classifier.cv_results_[
                    'mean_train_score'].mean())

            dict_metrics['validation_score'] = rounds(classifier.cv_results_[
                    'mean_test_score'].mean())

        dict_model[model_name] = [classifier, dict_metrics]

        if hasattr(self, 'models_container') and keep_track:

            self.models_container.append(dict_model)

        else:

            self.models_container = [dict_model]

        # Outcome Message
        print('\n')
        print(model_name + ' has been ingested into the models_container')

        if returns:

            return dict_model

    # Plot model confusion matrix
    def model_confusion_plot(self, model_name, root, path, filename,
                             figsize=(20, 12), bottom=0.3, top=0.9,
                             divpalette=sns.diverging_palette(240, 10, n=18),
                             setslot=None):
        """
         Confusion Matrix plot template.
         The method generates the confusion matrix visualization for
         the classifier model indicated in the model_name parameter,
         model which must have been loaded in the model_container by
         using the fit_ingestion method.
         The other method params are useful for setting the figure size,
         the margins and the colours.

         Parameters
         _ _ _ _ _
         model_name : String
             The name given to the model
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         filename : String
             The name of the file the figure is exported to
         figsize : tuple
             To set Matplotilib figure size
         bottom : float
             Bottom Figure Margin
         top : float
             Top Figure Margin
         divpalette : seaborn palette
             All the seaborn diverging palette are accepted
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.model_confusion_plot,
                                    dict_locals,
                                    setslot)
            return

        fig, ax = plt.subplots(figsize=figsize)

        fig.suptitle('Confusion Matrix :' + model_name,
                     weight=1000, fontstyle='oblique',
                     size=20, stretch=1000)

        ax = self._confusion_matrix_axes(ax, model_name, divpalette=divpalette)

        plt.subplots_adjust(bottom=bottom, top=top)

        plt.savefig(root + path + '\\' + filename + '.jpg', dpi=300)

        fig.clear()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + filename + '.jpg' +
              ' exported to ' + root + path)

    # Plot ROC Curve
    def model_roc_plot(self, model_name, root, path, filename, rate=0.1,
                       figsize=(20, 12), bottom=0.3, top=0.9,
                       setslot=None):
        """
         ROC CURVE plot template.
         The method generates the ROC Curve visualization for
         the classifier model indicated in the model_name parameter,
         model which must have been loaded in the model_container by
         using the fit_ingestion method.
         The other method params are useful for setting the plot level
         of detail, the figure size, the margins and the colours.
         Parameters
         _ _ _ _ _
         model_name : String
             The name of the model the roc curve is plotted from
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         filename : String
             The name of the file the figure is exported to
         rate : float
             The rate of the thresholds which the tpr and fpr are
             evaluated on. Greater values means a more stiff curve,
             a lower floating point, instead, will generate a more
             detailed and gentle curve.
         figsize : tuple
             To set Matplotilib figure size
         bottom : float
             Bottom Figure Margin
         top : float
             Top Figure Margin
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.model_roc_plot,
                                    dict_locals,
                                    setslot)
            return

        ROC_dataframe = self._roc_curve_data(model_name, rate=rate).copy()

        # Plot Figure
        fig, ax = plt.subplots(figsize=figsize)

        fig.suptitle('ROC Curve : ' + model_name,
                     weight=1000, fontstyle='oblique',
                     size=20, stretch=1000)

        ax = self._roc_curve_axes(ax, ROC_dataframe)

        plt.subplots_adjust(bottom=bottom, top=top)

        plt.savefig(root + path + '\\' + filename + '.jpg', dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + filename + '.jpg' +
              ' exported to ' + root + path)

    # Plot FEature Importance
    def model_features_plot(self, model_name, root, path, filename,
                            plot_threshold=0,
                            figsize=(20, 12), bottom=0.3, top=0.9,
                            rotation='vertical',
                            setslot=None):
        """
         Feature Importance plot template.
         The method generates the Feature Importance Ranking visualization
         for the classifier model indicated in the model_name parameter,
         model which must have been loaded in the model_container by
         using the fit_ingestion method.
         Notice that the plot is only available  for those
         classifiers which provide the feature_importance_ attribute.
         The other method params are useful for setting the number of
         features to display, the figure size, the margins and the x
         axis rotation.

         Parameters
         _ _ _ _ _
         model_name : String
             The name of the model the roc curve is plotted from
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         filename : String
             The name of the file the figure is exported to
         plot_threshold : float
             The plot_threshold param is to limit the features we
             want to display, by filtering those which do not get
             the limit set. Please, note that the threshold is applied
             on the normalized column, so take it into account
             when choosing the value.
             detailed and gentle curve.
         figsize : tuple
             To set Matplotilib figure size
         bottom : float
             Bottom Figure Margin
         top : float
             Top Figure Margin
         rotation : String
             Matplotilib parameter of set_xticklabels.
             Allowed values are:
                 'vertical': default
                 'horizontal'
                 angle in degrees : float from 0 to 1
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.model_features_plot,
                                    dict_locals,
                                    setslot)
            return

        feature_data = self._feature_importance_data(
                model_name, plot_threshold=0)

        # Plot Figure
        fig, ax = plt.subplots(figsize=figsize)

        fig.suptitle('Feature Importance Plot : ' + model_name,
                     weight=1000, fontstyle='oblique',
                     size=20, stretch=1000)

        ax = self._feature_importance_axes(ax, feature_data,
                                           rotation=rotation)

        plt.subplots_adjust(bottom=bottom, top=top)

        plt.savefig(root + path + '\\' + filename + '.jpg', dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + filename + '.jpg' +
              ' exported to ' + root + path)

    # Model Statistics and Scores Plot Template
    def model_plot(self, model_name, root,
                   path, filename,
                   figsize=(26, 14),
                   palette='dark', style='white',
                   left=0.03, right=0.97,
                   bottom=0.05, top=0.9,
                   wspace=0.25, hspace=0.22,
                   feat_xrotation='vertical',
                   feat_xthreshold=0,
                   roc_rate=0.1, roc_accuracy=True,
                   confusion_mtx_palette=None, fsize=14,
                   setslot=None):
        """
         Model Plots template.
         The method generates a figure with four subplots, allowing the
         user to have all the most relevant model metrics, scores and
         statistics in a single point of view.
         The model indicated in the model_name parameter must have
         been loaded in the model_container by using the fit_ingestion method.

         Subplot1 - Scores and Statistics

         Subplot2 - Confusion Matrix

         Subplot3 - ROC Curve

         Subplot4 - Feature Importance
         Notice that the plot is only available  for those
         classifiers which provide the feature_importance_ attribute.

         The method params enable the user to specify the filename and
         the path to export the figure, as long as many other params
         useful for adapting the plots and figure to specific
         visualization needs.

         Parameters
         _ _ _ _ _
         model_name : String
             The name of the model the plots are showed from
         root: String
            The root path the figure is exported to
         path : String
             The path the figure is exported to
         filename : String
             The name of the file the figure is exported to
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette to display
         style : string
             Seaborn style to display
         left : float
             Left Figure Margin
         rigth : float
             Right Figure Margin
         bottom : float
             Bottom Figure Margin
         top : float
             Top Figure Margin
         wspace : float
             Width space among the four subplots (horizontal)
         hspace : float
             High space among the four subplots (vertical)
         feat_xrotation : String
             Matplotilib parameter of set_xticklabels method for
             arranging thx x axis ticks rotation of the feature
             importance subplot.
             Allowed values are:
                 'vertical': default
                 'horizontal'
                 angle in degrees : float from 0 to 1
         feat_xthreshold : float
             The plot_threshold param is to limit the features we
             want to display in the feature importance subplot, by
             filtering those which do not get the limit set.
             Please, note that the threshold is applied on the
             normalized column, so take it into account when
             choosing the value.
         roc_rate : float
             The rate of the thresholds which the tpr and fpr are
             evaluated on. Applyies on the Roc Curve Subplot.
             Greater values means a more stiff curve,
             a lower floating point, instead, will generate a more
             detailed and gentle curve.
         roc_accuracy : Boolean
             if to plot on the right ROC Curve y axis the accuracy
             percentage in correspondence of each one of the
             threshold floating points
         confusion_mtx_palette : seaborn palette
             All the seaborn diverging palette are accepted
        fsize : integer
             Font size of text plots
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figure plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.model_features_plot,
                                    dict_locals,
                                    setslot)
            return

        dictm, classifier = self._get_model(model_name)[0:2]

        opts = self._options_plot_template(figsize=figsize,
                                           palette=palette,
                                           style=style,
                                           left=left, right=right,
                                           bottom=bottom, top=top,
                                           wspace=wspace, hspace=hspace,
                                           fsize=fsize)

        fig, gs, set_suptitle, set_plttitle, set_subtitle, \
            set_main, set_sub, cpalette, divpalette = opts

        c_one, c_two = cpalette[0], cpalette[5]

        if confusion_mtx_palette:
            divpalette = confusion_mtx_palette

        # Plotting Grid Space
        ax0, ax1 = [0, 1], [0, 1]

        ax0[0] = fig.add_subplot(gs[0, 0])
        ax0[0].set_title(model_name, set_plttitle)

        ax0[1] = fig.add_subplot(gs[0, 1])
        ax0[1].set_title('Confusion Matrix', set_plttitle)

        ax1[0] = fig.add_subplot(gs[1, 0])
        ax1[0].set_title('Roc Curve', set_plttitle)

        ax1[1] = fig.add_subplot(gs[1, 1])
        ax1[1].set_title('Feature Importance', set_plttitle)

        fig.suptitle('Model Statistics and Scores',
                     weight=1000, fontstyle='oblique', size=20,
                     stretch=1000, color=c_one)

        # Axis ax0[0]
        ax0[0].get_xaxis().set_ticks([])
        ax0[0].get_yaxis().set_ticks([])

        w1 = 0.30
        w2 = 0.60
        w3 = 0.90

        # Summary Statistics
        ax0[0].text(0.05, 0.90, 'Statistics of the test set ',
                    set_subtitle)

        ax0[0].text(w1, 0.80,
                    'Positive Cases : ' + '{:,d}'.format(dictm['ytest_1']),
                    set_main)
        ax0[0].text(w2, 0.80,
                    'Nagative Cases : ' + '{:,d}'.format(dictm['ytest_0']),
                    set_main)
        ax0[0].text(w3, 0.80,
                    'Avarage Priors : ' + '{0:.0%}'.format(
                            dictm['ytest_pmean']), set_main)

        # Test Scores
        ax0[0].text(0.05, 0.70, 'Model Scores : performed on the test set ',
                    set_subtitle)

        ax0[0].text(w1, 0.60, 'Model Score : ' +
                    str(dictm['model_score_name']).capitalize() +
                    ' {0:.0%}'.format(dictm['model_score']), set_main)

        ax0[0].text(w3, 0.60, 'Model Avarage Prediction : ' +
                    '{0:.0%}'.format(dictm['model_pmean']), set_main)

        ax0[0].text(w1, 0.50, 'Accuracy : ' + '{0:.0%}'.format(
                    dictm['score_accuracy']), set_main)

        ax0[0].text(w2, 0.50, 'Precision : ' + '{0:.0%}'.format(
                    dictm['score_precision']), set_main)

        ax0[0].text(w3, 0.50, 'Recall : ' + '{0:.0%}'.format(
                    dictm['score_recall']), set_main)

        ax0[0].text(w1, 0.40, 'AUC score : ' + '{0:.0%}'.format(
                    dictm['score_roc_auc']), set_main)

        ax0[0].text(w2, 0.40, 'F1 Score : ' + '{0:.0%}'.format(
                    dictm['score_f1']), set_main)

        ax0[0].text(w3, 0.40, 'Log Loss : ' + '{:.2f}'.format(
                    dictm['score_neg_log_loss']), set_main)

        # Grid Search Scores and Stats
        best_estimator = None
        if type(classifier) == GridSearchCV:

            best_estimator = classifier.best_estimator_

            ax0[0].text(0.05, 0.30, 'Grid Search Statistics ',
                        set_subtitle)

            ax0[0].text(w1, 0.25, 'Best Params', set_main)

            ax0[0].text(w2, 0.25, 'Training Score [avg] : ' +
                        '{0:.0%}'.format(dictm['train_score']), set_main)

            ax0[0].text(w3, 0.25, 'CV Score [avg] : ' +
                        '{0:.0%}'.format(dictm['validation_score']), set_main)

            ypos = 0.25
            for key, param in dictm['model_bestparms'].items():

                ypos = ypos - 0.04
                if ypos > 0:

                    ax0[0].text(0.30, ypos, key + str(param),
                                set_main)

        # Axis ax1[0]
        ax0[1] = self._confusion_matrix_axes(ax0[1], model_name,
                                             divpalette=divpalette)

        # Axis ax1[0]
        ax1[0] = self._roc_curve_axes(
                ax1[0], self._roc_curve_data(
                        model_name, rate=roc_rate).copy(),
                accuracy=roc_accuracy, color_accuracy=c_two)

        # Axis ax1[1]
        if hasattr(classifier, 'feature_importances_') or \
                hasattr(best_estimator, 'feature_importances_'):

            ax1[1] = self._feature_importance_axes(
                    ax1[1],
                    self._feature_importance_data(
                            model_name,
                            plot_threshold=feat_xthreshold),
                    rotation=feat_xrotation)

        else:

            dict_text = {'weight': 1000, 'fontstyle': 'oblique',
                         'size': 13, 'stretch': 1000, 'color': c_one,
                         'horizontalalignment': 'left'}

            ax1[1].text(0.05, 0.85, 'Feature Importance Plot absent ',
                        dict_text)

            ax1[1].text(0.05, 0.70, 'The model : ' + model_name.capitalize(),
                        dict_text)

            msg = 'Lacks the Feature_importances_ attribute'

            ax1[1].text(0.05, 0.55, msg,
                        dict_text)

            ax1[1].get_xaxis().set_ticks([])
            ax1[1].get_yaxis().set_ticks([])

        fig.tight_layout()

        plt.savefig(root + path + '\\' + filename + '.jpg', dpi=300,
                    bbox_inches='tight')

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + filename + '.jpg' +
              ' exported to ' + root + path)

        # Grid Search Figure II
        if type(classifier) == GridSearchCV:

            dic_pstat = self._grid_search_metrics(model_name)

            # Loop Over Parameters to create modular gridspec
            num = len(dic_pstat.keys())
            col = math.ceil(np.sqrt(num))
            row = math.ceil(num / col)

            opts = self._options_plot_template(figsize=figsize,
                                               palette=palette,
                                               style=style,
                                               left=left,
                                               right=right,
                                               bottom=bottom,
                                               top=top,
                                               wspace=wspace,
                                               hspace=hspace,
                                               row=row, col=col,
                                               fsize=fsize)

            fig, gs, set_suptitle, set_plttitle, set_subtitle, \
                set_main, set_sub, cpalette, divpalette = opts

            c_one, c_two, c_three = cpalette[0], cpalette[1], cpalette[2]

            fig.suptitle('Grid Search Parameters Plots',
                         weight=1000, fontstyle='oblique',
                         size=20, stretch=1000, color=c_one)

            # Figure with params related axes
            r = 0
            c = 0

            for idx, (key, value) in enumerate(dic_pstat.items()):

                def ax_plot(r, c):

                    ax = fig.add_subplot(gs[r, c])
                    ax.set_title(key, set_plttitle)

                    ax.plot(value.index,
                            value['mean_train_score'], color=c_two)
                    ax.plot(value.index,
                            value['mean_test_score'], color=c_three)

                    return ax

                if r < row:

                    ax_plot(r, c)
                    r = r + 1

                else:

                    c = c + 1
                    r = 0
                    ax_plot(r, c)
                    r = 1

            fig.tight_layout()

            plt.savefig(root + path + '\\' + filename + '_II.jpg', dpi=300,
                        bbox_inches='tight')

            plt.close()

            # Outcome Message
            print('\n')
            print('Matplotlib Figure ' + filename + '.jpg' +
                  ' exported to ' + root + path)

    # Plots Statistics and Scores for all trained models
    def models_plots(self, root, path,
                     figsize=(26, 14),
                     palette='dark', style='white',
                     left=0.03, right=0.97,
                     bottom=0.05, top=0.9,
                     wspace=0.25, hspace=0.22,
                     feat_xrotation='vertical',
                     feat_xthreshold=0,
                     roc_rate=0.1, roc_accuracy=True,
                     confusion_mtx_palette=None, fsize=14,
                     setslot=None):
        """
         Produce a Model Plots template for all the trained models
         hosted in the model container. See model_plot
         for details on the template and how to use the parameteres.
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.models_plots,
                                    dict_locals,
                                    setslot)
            return

        for dict_model in self.models_container:

            for x in dict_model.keys():

                model_name = x

                self.model_plot(
                        model_name, root,
                        path, model_name,
                        figsize=figsize,
                        palette=palette, style=style,
                        left=left, right=right,
                        bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace,
                        feat_xrotation=feat_xrotation,
                        feat_xthreshold=feat_xthreshold,
                        roc_rate=roc_rate,
                        roc_accuracy=roc_accuracy,
                        confusion_mtx_palette=confusion_mtx_palette,
                        fsize=fsize)

    # Inner Method for setting and exporting best model
    def model_set_best(self, mode='auto', model_name=None,
                       root=None,
                       path=None, filename=None,
                       returns=False,
                       setslot=None):
        """
         Method to retrieve the classifier which shows to return
         the best choosen score among trained models. It is possible
         to let the method compute the comparison based on a given score,
         or to simply give the model name we want to be set as the best.
         Choosing the best model, will make the ExplainableAI methods
         work on a default basis.

         Parameters
         _ _ _ _ _
         mode : string
             'auto': if to evaluate models on the deafult model score
              parameter. Note: it is a valid option only in the case
              where all the trained models provide the same default score.
              For sklearn classifiers, it is usually the accuracy, but if
              GridSearchCV  or another compatible not sklearn classifier
              have been trained, this conclusion is not so obvious.

             'manual': if to choose the model manually by giving the
              model name is wanted to be the best. In such case, provide
              the name in the model_name param.

              score name: if to evaluate models on a specified score.
              in this case, please provide the score_name.
              Allowed values are:
                score_accuracy
                score_precision
                score_recall
                score_roc_auc
                score_f1
                score_neg_log_loss
                model_score

         model_name : string
             If manual mode is choosen, it is the name of the trained model
             is wanted to be set as the best one.
         root: String
            The root path the figure is exported to
         path : string
             The path the best found classifier is exported to
         filename : string
             The filename the best found classifier is exported to
         returns : boolean
             If true return the model dictionary to the python session
        setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Fitted Classifier
             The best one found by the method
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.model_set_best,
                                    dict_locals,
                                    setslot)
            return

        self.best_model = {}

        if mode == 'auto':

            self.best_model = self._get_score('model_score')

        elif mode == 'manual':

            self.best_model[model_name] = self._get_model(model_name)[0:2]

        else:

            self.best_model = self._get_score(mode)

        print([x for x in self.best_model.keys()][0] +
              ' is set to be the best model')

        if path:
            obj_to_pickle(self.best_model, root + path, filename)

        if returns:

            return [x for x in self.best_model.values()][0][0]

# =============================================================================
# # <----- ----->
# # Explainable AI
# # <----- ----->
# =============================================================================

    # Inner Method to get shap values
    def _get_shap_values(self, model_name):
        """
         ....
         Parameters
         _ _ _ _ _
         model_name : String
             The name of the model the shap values are taken from
        """
        for dict_model in self.shap_container:

            if model_name == dict_model[0]:

                if type(dict_model[2][1][1]) == np.ndarray:

                    shap_values = dict_model[2][1]

                else:

                    shap_values = dict_model[2]

                expected_value = dict_model[1]

                return (expected_value, shap_values)

            else:
                pass

        print("Could not find the model in shapcontainer")
        raise KeyError

    # Explain the model and store shap values
    def shap_fit_ingestion(self, apply='best_model',
                           explainer='tree',
                           tree_model_output='margin',
                           tree_feature_perturbation='tree_path_dependent',
                           tree_approximate=False,
                           gradient_nsamples=200,
                           kernel_nsample=100,
                           keep_track=True,
                           returns=False,
                           setslot=None):
        """
         Based on the Shap Class the method apply the choosen shap
         explainer on the provided models. In default mode it explains the
         best pipiline model.
         For a wider explanation of the shap algoritms and theoretical
         approach, please refer to:
             https://github.com/slundberg/shap

         Parameters
         _ _ _ _ _
         apply : string or list
             if 'best_model' the model set as the best by the model_set_best
             method will be explained.
             ...
         explainer: string
             'tree': TreeExplainer
             'gradient': GradientExplainer
             'kernel': KernelExplainer
             Choose which of the explainer apply to the model
         tree_model_output: string
             "margin": raw output
             "probability": probability space
             "logloss": log base of the model loss function
             Inherited Shap.TreeExplainer parameter.
             Indeed, it only applies if explainer='tree'
             For the explanation on how to use it, please refers to
             the documentation provided by the command:
                 help(Shap.TreeExplainer)
         tree_feature_perturbation: string
             "interventional": breaks the dependencies between features
             "tree_path_dependent": default
             Inherited Shap.TreeExplainer parameter.
             Indeed, it only applies if explainer='tree'
             For the explanation on how to use it, please refers to
             the documentation provided by the command:
                 help(Shap.TreeExplainer)
         tree_approximate: boolean
             Inherited Shap.TreeExplainer.shap_values parameter.
             It approximate the Tree SHAP values, allowing a faster
             explainer execution.
         gradient_nsamples: integer
             Inherited Shap.GradientExplainer.shap_values parameter
             Indeed, it only applies if explainer='gradient'
             For the explanation on how to use it, please refers to
             the documentation provided by the command:
                 help(Shap.GradientExplainer)
         kernel_nsample: 'auto' or int
             Inherited Shap.KernelExplainer.shap_values parameter
             Number of times to re-evaluate the model when
             explaining each prediction.
             For the explanation on how to use it, please refers to
             the documentation provided by the command:
                 help(Shap.KernelExplainer)
         keep_track: boolean
             If true the fitted explainer is appended to the shap_container,
             otherwise the shap_container is reset and rubuild from scratch.
             By keeping track, the shap plot methods will be produced
             for all the explained models embedded in the container
             in a iterative loop.
         returns : boolean
             If true return the model dictionary to the python session
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         ....
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.shap_fit_ingestion,
                                    dict_locals,
                                    setslot)
            return

        def _shap_values(explainer, classifier):

            def _expected_value(explainer):
                try:
                    expected_value = explainer.expected_value[1]

                except Exception:
                    expected_value = explainer.expected_value

                return expected_value

            if explainer == 'tree':

                explainer = shap.TreeExplainer(
                        classifier,
                        data=self.df,
                        model_output=tree_model_output,
                        feature_perturbation=tree_feature_perturbation)

                return (_expected_value(explainer),
                        explainer.shap_values(self.dftest,
                                              approximate=tree_approximate))

            elif explainer == 'gradient':

                explainer = shap.GradientExplainer(classifier,
                                                   self.df)

                return (_expected_value(explainer),
                        explainer.shap_values(self.dftest,
                                              nsamples=gradient_nsamples))

            elif explainer == 'kernel':

                df_values = self.df.values

                shap_sampled = df_values[
                        np.random.choice(df_values.shape[0],
                                         kernel_nsample, replace=False)]

                df = pd.DataFrame(data=shap_sampled,
                                  columns=self.df.columns)

                explainer = shap.KernelExplainer(classifier.predict, df)

                return (_expected_value(explainer),
                        explainer.shap_values(self.dftest))

        if not (hasattr(self, 'shap_container') and keep_track):

            self.shap_container = list()

        if apply == 'best_model':

            key = [x for x in self.best_model.keys()][0]

            classifier = self.best_model[key][0]

            if type(classifier) == GridSearchCV:

                classifier = classifier.best_estimator_

            self.shap_container.append((key,) +
                                       _shap_values(explainer,
                                                    classifier))

        else:

            for dict_model in self.models_container:

                key = [x for x in dict_model.keys()][0]

                if apply == 'all' or (type(apply) == list and key in apply):

                    classifier = dict_model[key][0]

                    if type(classifier) == GridSearchCV:

                        classifier = classifier.best_estimator_

            self.shap_container.append((key,) +
                                       _shap_values(explainer,
                                                    classifier))

        if returns:

            return self.shap_container

    # Shap global plots
    def shap_global_plots(self, model_name, root, path,
                          figure_size=[20, 12],
                          palette='deep', style='white',
                          bottom=0.1, top=0.9,
                          left=0.2, right=0.9,
                          title_size=24,
                          summary_max_display=20,
                          decision_rec2sample=100,
                          decision_feat_range=slice(-1, -21, -1),
                          setslot=None):
        """
         By providing the model name of one of the models ingested in the
         fastlane container, the method produce three different shap plots,
         here called globals due to their ability to provide a
         global view of the model working way.
             summary_plot : The global importance plot of the model features,
             highlighting the positive or negative contribution direction.
             model_features_plot : the feature importance ranking plot,
             based on shap values in a standard display mode.
             decision_plot : given a sub-sample of records it highlights how
             each of the dataframe features impact on the model decision
             function, shifting in a or another way the sample prediction
             path.
         For a wider explanation, please refer to:
             https://github.com/slundberg/shap

         Parameters
         _ _ _ _ _
         model_name : String
             The name of the model the shap plots are originated by
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette name to display
         style : string
             Seaborn style name to display
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         title_size: integer
             Font size of figure title
         summary_max_display: integer
             Inherited Shap.summary_plot parameter, indeed it only
             applies to the summary plot
             It is the number of top features to include in the plot.
         decision_rec2sample: integer
             It is the number of sample to take in account when computing
             the decision plot. Please note that a too high value could
             crash the plot and make it harder to interpret.
         decision_feat_range: slice or range
             Inherited Shap.decision_plot parameter, indeed it only
             applies to the decision plot.
             It is the slice or range of features to plot after ordering
             features by feature_order. A step of 1 or None will display the
             features in ascending order. A step of -1 will display the
             features in descending order. If feature_display_range=None,
             slice(-1, -21, -1) is used.
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figures plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.shap_global_plots,
                                    dict_locals,
                                    setslot)
            return

        expected_value = self._get_shap_values(model_name)[0]
        shap_values = self._get_shap_values(model_name)[1]

        path = root + path + '\\' + model_name

        if not os.path.isdir(path):
            os.mkdir(path)

        sns.set(color_codes=True, style=style, palette=palette)

        cpalette = sns.color_palette()
        c_one = cpalette[0]

        # Summary Plot - Violin Fashion
        fig = plt.figure()

        fig.suptitle('Expailable AI - The Summary Plot',
                     weight=800, fontstyle='normal', size=title_size,
                     stretch=1000, horizontalalignment='center',
                     color=c_one)

        shap.summary_plot(shap_values,
                          self.dftest,
                          plot_size=figure_size,
                          plot_type='violin',
                          max_display=summary_max_display,
                          color=cpalette)

        plt.subplots_adjust(bottom=bottom, top=top,
                            left=left, right=right)

        plt.savefig(path + '\\' + 'summary_plot.jpg',
                    dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + 'summary_plot.jpg' +
              ' exported to ' + path)

        # Summary Plot - Feature Importance
        fig = plt.figure()

        fig.suptitle('Expailable AI - The Feature Importance Plot',
                     weight=800, fontstyle='normal', size=title_size,
                     stretch=1000, horizontalalignment='center',
                     color=c_one)

        shap.summary_plot(shap_values,
                          self.dftest,
                          plot_size=figure_size,
                          plot_type='bar',
                          max_display=summary_max_display,
                          color=cpalette)

        plt.subplots_adjust(bottom=bottom, top=top,
                            left=left, right=right)

        plt.savefig(path + '\\' + 'model_features_plot.jpg',
                    dpi=300)

        fig.clear()

        # Decision Plot
        shap_sampled = shap_values[
                np.random.choice(shap_values.shape[0],
                                 decision_rec2sample, replace=False)]

        fig = plt.figure()

        fig.suptitle('Expailable AI - The Decision Plot',
                     weight=800, fontstyle='normal', size=title_size,
                     stretch=1000, horizontalalignment='center',
                     color=c_one)

        shap.decision_plot(expected_value,
                           shap_sampled,
                           features=self.dftest.values,
                           feature_names=list(self.dftest.columns),
                           auto_size_plot=False,
                           feature_display_range=decision_feat_range)

        plt.gcf().set_size_inches(figure_size)

        plt.subplots_adjust(bottom=bottom, top=top,
                            left=left, right=right)

        plt.savefig(path + '\\' + 'decision_plot.jpg',
                    dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + 'decision_plot.jpg' +
              ' exported to ' + path)

    # Shap dependence plots for each df field
    def shap_dependence_plots(self, model_name, root, path,
                              interaction_index='auto',
                              figure_size=(20, 12),
                              palette='deep', style='white',
                              bottom=0.1, top=0.9,
                              left=0.2, right=0.9,
                              title_size=24,
                              alpha=1,
                              setslot=None):
        """
         By providing the model name of one of the models ingested in the
         fastlane container, the method produce a figure for each one of
         the dataframe column with a scatter plot inside, showing the
         correlation of the feature with the shap values.
         The plot also show the best interaction with another df feature.
         For a wider explanation, please refer to:
             https://github.com/slundberg/shap
         Parameters
         _ _ _ _ _
         model_name : String
             The name of the model the shap plots are originated by
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         interaction_index: "auto", None, int, or string
             Inherited Shap.dependence_plot parameter
             The index of the feature used to color the plot. The name of
             a feature can also be passed as a string. If "auto"
             then shap.common.approximate_interactions is used to pick what
             seems to be the strongest interaction.
         figsize : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette name to display
         style : string
             Seaborn style name to display
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         title_size: integer
             Font size of figure title
         alpha: float
             Inherited Shap.dependence_plot parameter
             The transparency of the data points (between 0 and 1).
             This can be useful to the show density of the data points
             when using a large dataset.
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figures plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.shap_dependence_plots,
                                    dict_locals,
                                    setslot)
            return

        shap_values = self._get_shap_values(model_name)[1]

        path = root + path + '\\' + model_name

        if not os.path.isdir(path):
            os.mkdir(path)

        sns.set(color_codes=True, style=style, palette=palette)

        cpalette = sns.color_palette()
        c_one = cpalette[0]

        for field in self.dftest.columns:

            fig = plt.figure()

            fig.suptitle('Expailable AI - The Dependence Plot: ' + field,
                         weight=800, fontstyle='normal', size=24,
                         stretch=1000, horizontalalignment='center',
                         color=c_one)

            shap.dependence_plot(field,
                                 shap_values,
                                 features=self.dftest.values,
                                 feature_names=list(self.dftest.columns),
                                 interaction_index=interaction_index,
                                 alpha=alpha)

            plt.gcf().set_size_inches(figure_size)

            plt.subplots_adjust(bottom=bottom, top=top,
                                left=left, right=right)

            plt.savefig(path + '\\depplot_' + field + '.jpg',
                        dpi=300)

            plt.close()

            # Outcome Message
            print('\n')
            print('Matplotlib Figure ' + field + '.jpg' +
                  ' exported to ' + path)

    # Shap Plots on single predictions
    def shap_one_plots(self, model_name, root, path, key,
                       figure_size=(22, 12),
                       palette='deep', style='white',
                       bottom=0.1, top=0.9,
                       left=0.2, right=0.9,
                       title_size=24,
                       waterf_max_display=15,
                       setslot=None):
        """
         By providing the model name of one of the models ingested in the
         fastlane container and a specifical dataframe key value, the method
         produce a shap force plot and a shap waterfall plot.
         Those figure let the analyst see and understand which and how
         the df features contributed to that individual prediction.
         Differently from the global plots, in this case we work on single
         records, detailing the analysis level granularity.

         Parameters
         _ _ _ _ _
         model_name : String
             The name of the model the shap plots are originated by
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         key: tuple
             Is the key table, in other words the dataframe index to pick
             for plotting. If it is a multiindex, please provide the
             full fields combination. In the case of a single index
             the tuple input format must be respected likewise.
         figure_size: : tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette name to display
         style : string
             Seaborn style name to display
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         title_size: integer
             Font size of figure title
         waterf_max_display: integer
             Inherited Shap.waterfall_plot parameter
             It is the maximum number of features to plot
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figures plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.shap_one_plots,
                                    dict_locals,
                                    setslot)
            return

        # Prepare Data
        dftest_dec = self._reset_dftest()

        rec2preditct = self.dftest.index.get_loc(key)

        classifier = self._get_model(model_name)[1]

        ypred = np.round(classifier.predict_proba(
                self.dftest.loc[[key]])[0, 1], 2)

        ytest = int(self.ytest.loc[key])

        expected_value = self._get_shap_values(model_name)[0]
        shap_values = self._get_shap_values(model_name)[1]

        path = root + path + '\\' + model_name

        if not os.path.isdir(path):
            os.mkdir(path)

        sns.set(color_codes=True, style=style,
                palette=palette)

        cpalette = sns.color_palette()
        c_one = cpalette[0]

        # Water Fall Plot
        fig = plt.figure()

        fig.suptitle('Expailable AI - The WaterFall Plot',
                     weight=800, fontstyle='normal', size=title_size,
                     stretch=1000, horizontalalignment='center',
                     color=c_one)

        set_main = {'weight': 800, 'fontstyle': 'normal', 'size': 15,
                    'stretch': 1000, 'horizontalalignment': 'right'}

        fig.text(0.12, 0.95, 'Model Predicion :', set_main)
        fig.text(0.13, 0.95, str(ypred))
        fig.text(0.12, 0.9, 'Ground Truth :', set_main)
        fig.text(0.13, 0.9, str(ytest))

        shap.waterfall_plot(expected_value,
                            shap_values[rec2preditct],
                            features=dftest_dec.values[rec2preditct],
                            feature_names=dftest_dec.columns,
                            show=False,
                            max_display=waterf_max_display)

        plt.gcf().set_size_inches(figure_size)

        plt.subplots_adjust(bottom=bottom, top=top,
                            left=left, right=right)

        plt.savefig(path + '\\' + str(key) + '_waterfall_plot.jpg',
                    dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + '_waterfall_plot.jpg' +
              ' exported to ' + path)

        # Force Plot
        fig = plt.figure()

        fig.suptitle('Expailable AI - The Force Plot',
                     weight=800, fontstyle='normal', size=title_size,
                     stretch=1000, horizontalalignment='center',
                     color=c_one)

        shap.force_plot(expected_value,
                        shap_values[rec2preditct],
                        features=np.round(self.dftest.values[
                                rec2preditct], 2),
                        feature_names=self.dftest.columns,
                        matplotlib=True,
                        link='logit')

        plt.gcf().set_size_inches(figure_size)

        plt.subplots_adjust(bottom=bottom, top=top-0.3,
                            left=left-0.15, right=right+0.05)

        plt.savefig(path + '\\' + str(key) + 'force_plot.jpg',
                    dpi=300)

        plt.close()

        # Outcome Message
        print('\n')
        print('Matplotlib Figure ' + str(key) + 'force_plot.jpg' +
              ' exported to ' + path)

    # Shap global plots for all the explained models
    def shap_models_global_plots(self, root, path,
                                 figure_size=[20, 12],
                                 palette='deep', style='white',
                                 bottom=0.1, top=0.9,
                                 left=0.2, right=0.9,
                                 title_size=24,
                                 summary_max_display=20,
                                 decision_rec2sample=100,
                                 decision_feat_range=slice(-1, -21, -1),
                                 setslot=None):
        """
         Produce global plots for all of the ingested explainers.
         Refer to the shap_global_plots method for more details.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         figure_size: tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette name to display
         style : string
             Seaborn style name to display
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         title_size: integer
             Font size of figure title
         summary_max_display: integer
             Inherited Shap.summary_plot parameter, indeed it only
             applies to the summary plot
             It is the number of top features to include in the plot.
         decision_rec2sample: integer
             It is the number of sample to take in account when computing
             the decision plot. Please note that a too high value could
             crash the plot and make it harder to interpret.
         decision_feat_range: slice or range
             Inherited Shap.decision_plot parameter, indeed it only
             applies to the decision plot.
             It is the slice or range of features to plot after ordering
             features by feature_order. A step of 1 or None will display the
             features in ascending order. A step of -1 will display the
             features in descending order. If feature_display_range=None,
             slice(-1, -21, -1) is used.
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figures plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.shap_models_global_plots,
                                    dict_locals,
                                    setslot)
            return

        for dict_model in self.shap_container:

            model_name = dict_model[0]

            self.shap_global_plots(
                    model_name, root, path,
                    figure_size=figure_size,
                    palette=palette, style=style,
                    bottom=bottom, top=top,
                    left=left, right=right,
                    title_size=title_size,
                    summary_max_display=summary_max_display,
                    decision_rec2sample=decision_rec2sample,
                    decision_feat_range=decision_feat_range)

    # Shap dependence plots for all the explained models
    def shap_models_dependence_plots(self, root, path,
                                     interaction_index='auto',
                                     figure_size=(20, 12),
                                     palette='deep', style='white',
                                     bottom=0.1, top=0.9,
                                     left=0.2, right=0.9,
                                     title_size=24,
                                     alpha=1,
                                     setslot=None):
        """
         Produce depende plots for all of the ingested explainers.
         Refer to the shap_dependence_plots method for more details.

         Parameters
         _ _ _ _ _
         root: String
            The root path the figure is exported to
         path: String
            The path the figure is exported to
         interaction_index: "auto", None, int, or string
             Inherited Shap.dependence_plot parameter
             The index of the feature used to color the plot. The name of
             a feature can also be passed as a string. If "auto"
             then shap.common.approximate_interactions is used to pick what
             seems to be the strongest interaction.
         figure_size: tuple
             To set Matplotilib figure size
         palette : string
             Seaborn palette name to display
         style : string
             Seaborn style name to display
         bottom : float
             Bottom Figure Margin for the four subplots
         top : float
             Top Figure Margin for the four subplots
         left : float
             Left Figure Margin for the four subplots
         right : float
             Right Figure Margin for the four subplots
         title_size: integer
             Font size of figure title
         alpha: float
             Inherited Shap.dependence_plot parameter
             The transparency of the data points (between 0 and 1).
             This can be useful to the show density of the data points
             when using a large dataset.
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Matplotlib figure
             Matplotlib figures plotted as mentioned in description
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.shap_models_dependence_plots,
                                    dict_locals,
                                    setslot)
            return

        for dict_model in self.shap_container:

            model_name = dict_model[0]

            self.shap_dependence_plots(
                    model_name, root, path,
                    interaction_index=interaction_index,
                    figure_size=figure_size,
                    palette=palette, style=style,
                    bottom=bottom, top=top,
                    left=left, right=right,
                    title_size=title_size,
                    alpha=alpha)

# =============================================================================
# # # <----- ----->
# # #  IN & OUT
# # # <----- ----->
# =============================================================================

    # Export Fastlane object to path
    def to_pickle(self, root, path, fname, mode='schedule',
                  setslot=None):
        """
         The method enable the user to export the entire BinaryLane object
         to the given path.
         In schedule mode, in other words, when the method will be part of a
         compiled pipiline and launched by the scheduler_exec, the scheduler
         array flow, where the scheduling information are stored, will not be
         exported, so that at re-importing time if needed you have to
         rebuilt it.

         Parameters
         _ _ _ _ _
         root: String
            The root path the pickle file is exported to
         path: String
            The path the pickle file is exported to
         mode : string
            schedule : scheduling mode
            standard : normal mode
         fname: string
            The name of the file the pickle is exported to
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         Pickle File
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.to_pickle,
                                    dict_locals,
                                    setslot)
            return

        if mode == 'schedule':

            schedule_argnames = self.schedule_argnames.copy()
            schedule_argvalues = self.schedule_argvalues.copy()
            schedule_functions = self.schedule_functions.copy()

            del self.schedule_argnames
            del self.schedule_argvalues
            del self.schedule_functions

        with open(root + path + '\\' + fname, 'wb') as file:

            pickle.dump(self, file,
                        protocol=pickle.HIGHEST_PROTOCOL)

            print('\n')
            print(fname + ' exported to ' + root + path)

        if mode == 'schedule':

            self.schedule_argnames = schedule_argnames
            self.schedule_argvalues = schedule_argvalues
            self.schedule_functions = schedule_functions

    # Get DataFrame
    def get(self, dict_cols=False, apply2test=False,
            setslot=None):
        """
         It is the enabler of free analysis.
         Moving aways, from the pipiline standardized road, the method
         let the user get the dataframes and dictionary columns out of
         the BinaryLane to the python session.
         In this way, the pipiline could be implemented until the step
         considered to be appropriate for own project purposes, and eventually
         recovered in a further moment. In the latter case by using the load
         method.
         The main idea, here, is to give the developer a tool that
         could speed its analysis and operation freeing up his-her time by
         providing pre-made commonly consumed machine learning task objects
         and methods but without depriving his-her freedom of customization and
         fine tunings.

         Parameters
         _ _ _ _ _
         dict_globals: dictionary
             If to also get the df colums dictionary out of the pipiline
         apply2test : Boolean
             If to also get test set out of the pipiline
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         FastLane dataframe
         FastLane test dataframe if apply2test == True
         FastLane dictionary of columns if dict_cols == True
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.get,
                                    dict_locals,
                                    setslot)
            return

        if dict_cols:

            if apply2test:

                return self.df.copy(), self.dict_cols.copy(), \
                    self.dftest.copy()

            return self.df.copy(), self.dict_cols.copy()

        if apply2test:

            return self.df.copy(), self.dftest.copy()

        return self.df.copy()

    # Get DataFrame
    def load(self, df, inp_dict_cols=None, apply2test=False, dftest=None,
             setslot=None):
        """
         It is the enabler of taking back the standard pipiline from
         free analysis and custom operations, so that one can get in and out
         of the pipiline whenever desired as long as the consistency Binary-
         Lane checks are respected.
         The main idea, here, is to give the developer a tool that
         could speed its analysis and operation freeing up his-her time by
         providing pre-made commonly consumed machine learning task objects
         and methods but without depriving his-her freedom of customization and
         fine tunings.

         Parameters
         _ _ _ _ _
         df : Pandas DataFrame
             The input table
         inp_dict_cols : dictionary
             The input Dictionary containing keys: 'Pandas Data Type Group',
             values: 'list of df column names'
         dftest : Pandas DataFrame
             The eventual test data frame (if exist)
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         df dataframe
         dftest dataframe if apply2test == True
         dictionary of columns if dict_cols == True
             self instance setting : df, [dftest, [dict_cols]]
        """
        if setslot:

            dict_locals = locals()
            self._set_schedule_flow(self.load,
                                    dict_locals,
                                    setslot)
            return

        pandas_dtypes = ['fmtcategory', 'fmtordcategory',
                         'fmtint', 'fmtfloat', 'fmtdatetime']

        for pdtype in pandas_dtypes:

            try:

                inp_dict_cols[pdtype]

                if pdtype == 'fmtcategory':
                    self.fmtcategory = self.fmtcategory + \
                        inp_dict_cols[pdtype]

                if pdtype == 'fmtordcategory':
                    self.fmtordcategory = self.fmtordcategory + \
                        inp_dict_cols[pdtype]

                if pdtype == 'fmtint':
                    self.fmtint = self.fmtint + \
                        inp_dict_cols[pdtype]

                if pdtype == 'fmtfloat':
                    self.fmtfloat = self.fmtfloat + \
                        inp_dict_cols[pdtype]

                if pdtype == 'fmtdatetime':
                    self.fmtdatetime = self.fmtdatetime + \
                        inp_dict_cols[pdtype]

            except Exception:
                continue

        if inp_dict_cols:

            for key, value in inp_dict_cols.items():

                if key in pandas_dtypes:

                    if key in self.dict_cols.keys():

                        self.dict_cols[key] = self.dict_cols[key] + value

                    else:

                        self.dict_cols[key] = value

        self.df = df.copy()

        if apply2test:

            self.df, self.dftest = df.copy(), dftest.copy()

# =============================================================================
# # <----- ----->
# # Scheduler
# # <----- ----->
# =============================================================================

    # Inner Method to dinamically get function arguments
    def _get_schedule_funcargs(self, function, dict_locals):
        """
         Take out of the given function its parameters, returning the
         list of the param names and that one of param values.

         Parameters
         _ _ _ _ _
         function: Python Function
             The python function the arguments are taken from
         dict_locals: dictionary
             Dictionary of funtion local variabels.

         Returns
         _ _ _ _ _
         listnames : List of function argument names
         listvalues : List of function argument values
        """
        if 'self' in dict_locals.keys():

            del dict_locals['self']

        if 'setslot' in dict_locals.keys():

            del dict_locals['setslot']

        listnames = []
        listvalues = []

        for key, value in dict_locals.items():

            listnames.append(key)
            listvalues.append(value)

        return listnames, listvalues

    # Inner Method to set the scheduled flow
    def _set_schedule_flow(self, function, dict_locals, setslot):
        """
         Feed the schedule container with the new provided function.
         If it is the first round the container is created, differently
         the given function is appended to the existing one.

         Parameters
         _ _ _ _ _
         function: Python Function
             The python function the arguments are taken from
         dict_locals: dictionary
             Dictionary of funtion local variabels.
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         schedule_argnames : List of List function argument names
         schedule_argvalues : List of List function argument values
         schedule_functions : List of functions
             self instance setting
        """
        if setslot is not None:

            listnames, listvalues = self._get_schedule_funcargs(function,
                                                                dict_locals)

            self.schedule_argnames, \
                self.schedule_argvalues, \
                self.schedule_functions = \
                self._set_schedule_container(listnames, listvalues,
                                             function, setslot)

    # Inner Method to create the scheduled flow
    def _schedule_container_create(self, listnames, listvalues,
                                   function, setslot):
        """
         Create the schedule container where to store the compiled function and
         its parameters.

         Parameters
         _ _ _ _ _
         listnames: list
             List of function argument names
         listvalues: list
             List of function argument values
         function: python function
             the function to compile in the schedule container
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         schedule_argnames : List of List function argument names
         schedule_argvalues : List of List function argument values
         schedule_functions : List of functions
        """
        if setslot == -1:
            setslot = 1

        schedule_argnames = [0]
        schedule_argvalues = [0]
        schedule_functions = [0]

        try:
            schedule_argnames[setslot-1] = listnames
            schedule_argvalues[setslot-1] = listvalues
            schedule_functions[setslot-1] = function

        except IndexError:
            print('\n')
            print('Flow Step out of index: ',
                  'Please set correctly the setslot parameter')

        return schedule_argnames, schedule_argvalues, schedule_functions

    # Inner Method to add a function to the scheduled flow
    def _schedule_container_append(self, listnames, listvalues,
                                   function, setslot):
        """
         Append the given function and its parameters to the schedule array.

         Parameters
         _ _ _ _ _
         listnames: list
             List of function argument names
         listvalues: list
             List of function argument values
         function: python function
             the function to compile in the schedule container
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         schedule_argnames : List of List function argument names
         schedule_argvalues : List of List function argument values
         schedule_functions : List of functions
        """
        if setslot == -1:
            setslot = len(self.schedule_argvalues) + 1

        if len(self.schedule_argvalues) == setslot - 1:

            self.schedule_argnames.append(0)
            self.schedule_argvalues.append(0)
            self.schedule_functions.append(0)

        try:

            self.schedule_argnames[setslot-1] = listnames
            self.schedule_argvalues[setslot-1] = listvalues
            self.schedule_functions[setslot-1] = function

        except IndexError:
            print('\n')
            print('Flow Step out of index: ',
                  'Please set correctly the setslot parameter')

        return self.schedule_argnames, self.schedule_argvalues, \
            self.schedule_functions

    # Inner Method to set the scheduled flow
    def _set_schedule_container(self, listnames, listvalues,
                                function, setslot):
        """
         It is the method that decide where to direct the compilation request,
         to the schedule container creation method or to the schedule
         append one.

         Parameters
         _ _ _ _ _
         listnames: list
             List of function argument names
         listvalues: list
             List of function argument values
         function: python function
             the function to compile in the schedule container
         setslot : integer
             Please, refer to the constructor method for a wider
             parameter description.

         Returns
         _ _ _ _ _
         schedule_argnames : List of List function argument names
         schedule_argvalues : List of List function argument values
         schedule_functions : List of functions
        """
        if not hasattr(self, 'schedule_argvalues'):

            schedule_argnames, schedule_argvalues, schedule_functions = \
                self._schedule_container_create(listnames, listvalues,
                                                function, setslot)

        else:

            schedule_argnames, schedule_argvalues, schedule_functions = \
                self._schedule_container_append(listnames, listvalues,
                                                function, setslot)

        return schedule_argnames, schedule_argvalues, schedule_functions

    # scheduler
    def scheduler_exec(self, dict_globals,
                       start=None, end=None):
        """
         It is the core method for automating the pipiline.
         Indeed, the BinaryLane can be used by the developer in a dual mode:
             - Interactive: the object methods are implemented step by step and
             results returned to the developer python session (returns=True).
             - Compilied: the pipiline is created in a .py file all at once,
             with each one of the used methods having the setslot parameter
             set, and executed in a second time, in separate python file.
             The latter is the working area of the scheduler_exec.
        Once the pipiline is created in a first .py file, the resulting
        binarylane object can be imported in a second python file,
        by using the commonly used import python function.
        In this way the scheduler_exec can be applied upon the just import
        binarylane object.
        Before execution the user can creare a dictionary of global variables,
        whose goal is to enable the changing-updating of compilied method
        parameter at execution time. This feature may ve very effective if
        we want to run more execution pipiline in the same session, and we
        want some of the method parameters to be set with specific values
        for each one of the execution. The main example is the root path that
        can be changed before each one of the scheduler_exec callings,
        so that the plots or pickle objects can be directed and exported in
        different and execution related locations.
        Furthemore, the compiled pipiline can be executed in n different times,
        by limiting the step by step operation to only those included between
        the starting and ending positions provided.
        This offer the developer flexibility to introduce new custom operations
        among the several scheduler_executions.
        This feature is enhanced by updating\replacing specific compiled
        operation steps at execution time, by implementing a binarylane method
        with  the setslot parameter established to the that specific step
        posiion. For example, if I want to change the balancing method and I
        know it has the fifth pipiline position, before calling the sceduler
        exec a can implement a df_balance method with the parameter setslot
        fixed to 5.

         Parameters
         _ _ _ _ _
         dict_globals: dictionary
             Dictionary of global variables
         start: integer
             The starting scheduler position
         end: integer
             The ending scheduler position
         Returns
         Schedule work flow execution
         ....
        """
        for idx, (names, params, function) in enumerate(zip(
                                self.schedule_argnames,
                                self.schedule_argvalues,
                                self.schedule_functions)):

            idx = idx + 1

            if start is None and end is None:

                for name in names:

                    if name in dict_globals.keys():

                        new_param = dict_globals[name]

                        pos = names.index(name)

                        params[pos] = new_param

                function(*params)

            elif (idx >= start and idx <= end):

                for name in names:

                    if name in dict_globals.keys():

                        new_param = dict_globals[name]

                        pos = names.index(name)

                        params[pos] = new_param

                function(*params)

    # Reset Scheduler
    def scheduler_reset(self):
        """
         Drop schedule arrays
        """
        if hasattr(self, 'schedule_argvalues'):

            del self.schedule_argnames
            del self.schedule_argvalues
            del self.schedule_functions

    # Reset Object Pre Reschedule
    def scheduler_restart(self, df, dict_cols):
        """
         Clean FastLane object and import df and dictcols,
         so that the compiler is ready for another execution into
         the same python session.
         It is mandatory to import df, dictcols in the execution session
         and use this method before schedule_exec, if 2 or more execution
         are made at once.
        """
        dict_obj = self.__dict__.copy()

        for key in dict_obj.keys():

            if key not in ['schedule_argnames',
                           'schedule_argvalues',
                           'schedule_functions']:

                delattr(self, key)

        self.schedule_argvalues[0][0] = df.copy()
        self.schedule_argvalues[0][2] = dict_cols.copy()
