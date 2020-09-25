import pickle
import pandas_profiling as pp
import pandas as pd
import contextlib
import saspy
import re

"""
 <<< open_sas() >>>
     Sas Context Manager
"""


@contextlib.contextmanager
def open_sas():

    sas = saspy.SASsession(cfgname='winiomprod')

    print('The SAS Working Path is ' + str(sas.workpath))
    print('The SASAssigned librefs are ' + str(sas.assigned_librefs()))

    yield sas

    sas.endsas()


"""
 <<< obj_to_pickle(obj_to_export, path, fname)>>>
 <<< obj_from_pickle(path, fname)>>>
    Export the python objet to the specified file in the passed path
    Import the pickle file from the specified path
"""


def obj_to_pickle(obj_to_export, path, fname):
    with open(path + '\\' + fname, 'wb') as file:
        pickle.dump(obj_to_export, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(fname + ' exported to ' + path)


def obj_from_pickle(path, fname):
    with open(path + '\\' + fname, 'rb') as file:
        obj_imported = pickle.load(file)
        print(fname + ' imported from ' + path)
        return obj_imported


"""
 <<< import_sas_contents(tables, librefs, kwargs)>>>
 Import a list of Sas Tables in a Dictionary

 Parameters
 _ _ _ _ _
 sas       : SasConnection Object
 tables    : List
           List of String SAS Tables Names
 librefs   : list
           List of String Librefs in which Sas Tables are hold
 kwargs    : list of SAS options dictionries
           List of Dictionaries of SAS options (ex: obs, keep, drop, ...)

 Returns
 _ _ _ _ _
 Dictionary of Key (Table Name) : Values (2 arg Tuple)
 Key       : Name of the SAS table
 Values    : the 1Th Tuple Argument is Sas Table Name
           : the 2th Tuple Argument is a dictionary of column type lists
           The colomn types are inferred from the SAS Table Labels and grouped
           in list conteining the column variable names, label related
           If no Label is provided an empty Lisnan is created
"""


def import_sas_contents(sas, tables, librefs, kwargs):

    dict_ds2df = dict()

    for table, libref, kwargs in zip(tables, librefs, kwargs):

        print(table)
        print(kwargs)
        ds = sas.sasdata(table, libref, 'Pandas', kwargs)
        ds_col = ds.columnInfo().loc[ds.columnInfo()['Label'] != 'No',
                                     ['Variable', 'Label']]

        df = ds.to_df(method='CSV')[ds_col['Variable']]

        dict_cat = dict()

        for label in ds_col['Label'].unique():
            lst = ds_col.loc[ds_col['Label'] == label, 'Variable'].to_list()
            key = 'lst' + str(label)

            dict_cat[key] = lst

        dict_ds2df[table] = (df, dict_cat)

    return dict_ds2df


"""
 <<< dict2dict_pdtypes(dict_lst, dict2fmt = {}, Update = True)>>>
 Assigns Lists hold in a Dictionary to formatted grouped list

 Parameters
 _ _ _ _ _
 dict_lst  : Dictionary
           Input Dictionary contanining pair of Key "List Names" :
               Values "List values"

 dict2fmt  : Dictionary
           Mapping Dictionary containing pair of Key "List Names" :
               Values "Names of the lists to be assigned"

 Update    : Boolean
           If True dict2fmt is updated with the new entries, otherwise
               the default is kept.

           If False the dictionary will be build on scratch by using the input:
                    "lstKey"                : "key",
                    "lstContinuousNumber"   : "fmtfloat",
                    "lstOrdinalNumber"      : "fmtint",
                    "lstDummy"              : "fmtint",
                    "lstCategory"           : "fmtcategory",
                    "lstOrdinalCategory"    : "fmtordcategory",
                    "lstDate"               : "fmtdatetime"
 Returns
 _ _ _ _ _

 Dictionary (containing Six Lists) :
     key, fmtint, fmtfloat, fmtobject, fmtcategory, fmtdatetime

 Description
 _ _ _ _ _

 The Function gets a set of lists containing Column names to be mapped with a
 group of pre defined format list which are Pandas Data Type related.

 Using this function might help providing prepared data to the DataFrame to be
 formatted. The Function has its default mapping dictionary, which might be
 updated with new entries, inserted in the dict2fmt dict.

 If Default parameters are kept, no mapping dataframe passed, the default one
 will be used. If one of the 6 resulting list is not used by the function,
 the result will be a list with no values.
"""


def dict2dict_pdtypes(dict_lst, dict2fmt_update={}, Update=False):

    dict2fmt = {
            'lstKey': 'key',
            'lstContinuousNumber': 'fmtfloat',
            'lstOrdinalNumber': 'fmtint',
            'lstDummy': 'fmtcategory',
            'lstCategory': 'fmtcategory',
            'lstOrdinalCategory': 'fmtordcategory',
            'lstDate': 'fmtdatetime'
            }

    if Update:

        if not dict2fmt_update:
            dict2fmt_update = dict()

        dict2fmt.update(dict2fmt_update)

    dict2_lstfmt = {}
    keytbl = []
    fmtint, fmtfloat, fmtcategory, fmtordcategory, fmtdatetime = [], [], [], \
                                                                 [], []

    for key, value in dict_lst.items():

        try:
            fmtlist = dict2fmt[key]

        except:
            continue

        if fmtlist == 'key':
            dict2_lstfmt['key'] = keytbl + value
        if fmtlist == 'fmtint':
            fmtint = fmtint + value
            dict2_lstfmt['fmtint'] = fmtint
        if fmtlist == 'fmtfloat':
            fmtfloat = fmtfloat + value
            dict2_lstfmt['fmtfloat'] = fmtfloat
        if fmtlist == 'fmtcategory':
            fmtcategory = fmtcategory + value
            dict2_lstfmt['fmtcategory'] = fmtcategory
        if fmtlist == 'fmtordcategory':
            fmtordcategory = fmtordcategory + value
            dict2_lstfmt['fmtordcategory'] = fmtordcategory
        if fmtlist == 'fmtdatetime':
            fmtdatetime = fmtdatetime + value
            dict2_lstfmt['fmtdatetime'] = fmtdatetime

        for key, value in dict2_lstfmt.items():
            value = list(dict.fromkeys(value))
            dict2_lstfmt[key] = value

    return dict2_lstfmt


"""
 <<< ddfs2ddfs_pdtypes(dict_df, dict2fmt={}, Update=False)>>>

 Assigns Df Col Lists to pdtypes grouped list

 Parameters
 _ _ _ _ _

 dict_df : Dictionray
         The input Dictionary containing Keys:"DataFrame Name", values:"Tuples"
         with in Index 0 the DataFrame itself,in Index 1 the second dictionary:
             containing Keys:"Type List Name", Values: "Data Frame Colum names
             which have to be formatted"

 dict2fmt : Dictionary
          Please refer to dict2fmtlst function

 Update   : Boolean
          Please refer to dict2fmtlst function

 Returns
 _ _ _ _ _

 Dictionary:

     3 args dictiorary -
     Input Dictionary : df, orig col lists +
     Dictionary Pdtype List Grouped
"""


def ddfs2ddfs_pdtypes(dict_df, dict2fmt_update={}, Update=False):

    dict_new = dict()

    for key in dict_df:

        df = dict_df[key][0].copy()
        dict_ = dict_df[key][1]

        dict_pdtype = dict2dict_pdtypes(dict_, dict2fmt_update, Update)

        dict_new[key] = (df, dict_, dict_pdtype)

    return dict_new


"""
 <<< ddfs2dict_warnings(dict_df, dict2fmt={}, Update=False)>>>
 Controlling DataFrame Data Type Columns and return warnings

 Parameters
 _ _ _ _ _

 dict_df : Dictionray
         The input Dictionary containing Keys:"DataFrame Name", values:"Tuples"
         with in Index 0 the DataFrame itself,in Index 1 the second dictionary:
             containing Keys:"Type List Name", Values: "Data Frame Colum names
             which have to be formatted"

 dict2fmt : Dictionary
          Please refer to dict2fmtlst function

 Update   : Boolean
          Please refer to dict2fmtlst function

 Returns
 _ _ _ _ _

 A Dictionary containing warnings grouped for DataFrame and Pandas Data Type
"""


def ddfs2dict_warnings(dict_df, dict2fmt_update={}, Update=False):

    dict_wrn = dict()

    # Iterate over each DataFrame
    for key in dict_df:

        # Get DataFrame and dict of Pandas data type groups
        df = dict_df[key][0].copy()
        dict_lstfmt = dict2dict_pdtypes(dict_df[key][1],
                                        dict2fmt_update, Update)
        dict_wrn[key] = dict()

        # Get Key : Pandas Dtype groups and values : list of fields
        for key_lst, value_lst in dict_lstfmt.items():

            if key_lst == 'key':

                dict_wrn[key][key_lst] = dict()

                arr_dup = df.set_index(dict_lstfmt['key']).index.duplicated()

                dict_wrn[key][key_lst] = 'Key duplicate values. ' + \
                    str(arr_dup.sum()) + ' cases'

            if key_lst == 'fmtint':

                dict_wrn[key][key_lst] = dict()

                # Iterate each series of the df
                for field in value_lst:

                    dict_wrn[key][key_lst][field] = dict()

                    if df[field].dtype == 'float64':

                        sr_bool = df[field].apply(lambda x:
                                                  bool(re.fullmatch(
                                                          r'^\-?\d+(\.\d+)?',
                                                          str(x))))
                    else:

                        sr_bool = df[field].apply(lambda x:
                                                  bool(re.fullmatch(r'^\-?\d+',
                                                                    str(x))))

                    if (sr_bool == False).any():

                        dict_values = dict(df[field][sr_bool == False].
                                           value_counts(dropna=False))

                        dict_wrn[key][key_lst][field] = dict_values

            if key_lst == 'fmtfloat':

                dict_wrn[key][key_lst] = dict()

                # Iterate each series of the df
                for field in value_lst:

                    dict_wrn[key][key_lst][field] = dict()

                    sr_bool = df[field].apply(lambda x:
                                              bool(re.fullmatch(
                                                      r'^\-?\d+(\.\d+)?',
                                                      str(x))))

                    if (sr_bool == False).any():

                        dict_values = dict(df[field][sr_bool == False].
                                           value_counts())

                        dict_wrn[key][key_lst][field] = dict_values

            if key_lst == 'fmtordcategory':

                dict_wrn[key][key_lst] = dict()

                # Iterate each series of the df
                for field in value_lst:

                    dict_wrn[key][key_lst][field] = dict()

                    sr_bool = df[field].apply(lambda x:
                                              bool(re.fullmatch(
                                                      r'^\-?\d+(\.\d+)?',
                                                      str(x))))

                    if (sr_bool == False).any():

                        dict_values = dict(df[field][sr_bool == False].
                                           value_counts(dropna=False))

                        dict_wrn[key][key_lst][field] = dict_values

            if key_lst == 'fmtdatetime':

                dict_wrn[key][key_lst] = dict()

                # Iterate each series of the df
                for field in value_lst:

                    dict_wrn[key][key_lst][field] = dict()

                    sr_bool = df[field].apply(
                            lambda x: bool(re.fullmatch(
                                    r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}',
                                    str(x))))

                    if (sr_bool == False).any():

                        dict_values = dict(df[field][sr_bool == False].
                                           value_counts(dropna=False))

                        dict_wrn[key][key_lst][field] = dict_values

    return dict_wrn


"""
 <<< df_engeneering(df, dict_pdtypes, mapfunc)>>>

 in an ApplyMap Fashion transforms, replacing values, the input df columns

 Parameters
 _ _ _ _ _

 df : DataFrame
    Input DataFrame to be transformed

 dict_pdtypes : Dictionary
    Dictionary of Key: Pandas Dtypes, values: List of columns

 mapfunc : Tuple
    [0] Pandas Dtype to be mapped with key dict_pdtypes
    [1] Lambda Function to be applied in Applymap (scalar func)

  Returns
 _ _ _ _ _

 Transformed Data Frame
"""


def df_engeneering(df, dict_pdtypes, mapfunc):

    for func in mapfunc:

        col_pdtypes = dict_pdtypes[func[0]]

        col_origtypes = df[col_pdtypes].dtypes

        # Loop Over original df dtype returing its original dtype
        for orig_dtype in col_origtypes.unique():

            cols_origtype = [x for x in col_origtypes[
                    col_origtypes == orig_dtype].index]

            df[cols_origtype] = df[cols_origtype].applymap(
                    func[1]).astype(orig_dtype)

    return df.copy()


"""
 <<< ddfs_engeneering(dict_df, df_mapfunc)>>>

  In an ApplyMap Fashion transforms, replacing values, the input
  dictionary dataframe columns

 Parameters
 _ _ _ _ _

  dict_df : Dictionray (3args)
         The input Dictionary containing Keys:"DataFrame Name", values:"Tuples"
         with in Index 0 the DataFrame itself,
              in Index 1 the second dictionary:containing Keys: Type List Name
              Values: "Data Frame Colum names which have to be formatted"
              in Index 2 the third dictionary:containing Keys: Pdtype List Name
              Values: "Data Frame Colum names which have been formatted"
 Returns
 _ _ _ _ _

 Dictionary of DataFrames (3 args). For each:

     [0] Transformed DataFrame
     [1] Orig col Lists:
     [2] Pdtype cols
"""


def ddfs_engeneering(dict_df, df_mapfunc):

    for idx, key in enumerate(dict_df):

        try:
            df = dict_df[key][0].copy()
            dict_pdtypes = dict_df[key][2]

            # Build the df specific func tuple
            mapfunc = tuple()
            for pos in df_mapfunc:

                if pos[0] is None or pos[0] == key:

                    func = [pos[1], pos[2]],
                    mapfunc = mapfunc + func

            df_new = df_engeneering(df, dict_pdtypes, mapfunc)

            dict_df[key] = (df_new, dict_df[key][1], dict_pdtypes)

        except Exception:
            continue

    return dict_df


"""
 <<< ddfs_apply_pdtype(dict_df, dict_pdtypes) >>>

 Formatting a dict of DataFrames based on a Dictionary of pandas data type
 column list groups

 Parameters
 _ _ _ _ _

 dict_df : Dictionray (3args)
         The input Dictionary containing Keys:"DataFrame Name", values:"Tuples"
         with in Index 0 the DataFrame itself,
              in Index 1 the second dictionary:containing Keys: Type List Name
              Values: "Data Frame Colum names which have to be formatted"
              in Index 2 the third dictionary:containing Keys: Pdtype List Name
              Values: "Data Frame Colum names which have been formatted"

 dict_pdtypes : Dictionary
         Dictionary of Key: Pandas Dtypes, values: List of df columns

 Returns
 _ _ _ _ _

 Dictionary (3args)
 The input dictionary with formatted DataFrames in it
"""


def ddfs_apply_pdtype(dict_df):

    dict_mapp = {'fmtint': 'int', 'fmtfloat': 'float', 'fmtcategory': 'object',
                 'fmtordcategory': 'category', 'fmtdatetime': 'datetime64'}

    for key in dict_df:

        df = dict_df[key][0]
        dict_pdtypes = dict_df[key][2]

        for key_pdtype, df_cols in dict_pdtypes.items():

            if key_pdtype == 'key':
                df.set_index(df_cols, inplace=True, drop=True)

            else:
                pformat = dict_mapp[key_pdtype]

                for col in df_cols:

                    try:
                        df[col] = df[col].astype(pformat)

                    except Exception:
                        print(col + ' was not converted in ' + pformat +
                              ' due to conversion problems')

                if key_pdtype == 'key':
                    df.set_index(df_cols, inplace=True, drop=True)

    return dict_df


"""
 <<< ddicts2dict_of_columns(ddicts, idx_pos=[1, 2]) >>>

 Grouping in a single dictionary the groups of colum lists of n dataframes

 Parameters
 _ _ _ _ _

 ddicts : Dictionary
        The input Dictionary containing Keys:"DataFrame Name", values:"Tuple"
        with in Index 0 the DataFrame itself,
              in Index 1 to Index n the other dictionaries containing -
              Keys: Type List Name, Values: Data Frame Colum names

 idx_pos : List
        The List containing the Tuple positions in which the dictionary
        of group columns is located

 Returns
 _ _ _ _ _

 Dictionary

 Keys: Grouped Type List Col Name
 Values : Col lists

"""


def ddicts2dict_of_columns(ddicts, idx_pos=[1, 2]):

    dict_of_columns = dict()

    for idx in idx_pos:

        for key in ddicts:

            for groupcol, columns in ddicts[key][idx].items():

                try:
                    base_cols = dict_of_columns[groupcol]
                    dict_of_columns[groupcol] = base_cols + columns

                except Exception:

                    dict_of_columns[groupcol] = columns

        for key, value in dict_of_columns.items():

            dict_of_columns[key] = list(dict.fromkeys(value))

    return dict_of_columns


"""
 <<< ddfs2df_append(dict_df, tuple_of_joins, df_return=True, df_2pickle=False,
                   drop_index=False)>>>

 Concatenating DataFrames on an User defined input Tuple Join sequence,
 returning df on the session and\or on pickle file

 Parameters
 _ _ _ _ _

 dict_df : Dictionray
         The input Dictionary containing Keys:"DataFrame Name", values:"Tuples"
         with in Index 0 the DataFrame itself

 tuple_of_joins : Tuple
         A tuple containing tuples whose index structure is:
            element 1 : dataframe name
            element 2 : left table join condition
            element 3 : right table join condition
            element 4 : join condition
            element 4 : validation condition

 df_return : Boolean
           If to return a dataframe in the python session

 df_2pickle : Booelan
            If to return a pickle file containing the df. In such a case,
            indicate the path the pickle file is wanted to be stored

 Returns
 _ _ _ _ _

 1 Pandas DataFrame,
 2 Pickle df file,
"""


def ddfs2df_append(dict_df, tuple_of_joins, df_return=True, df_2pickle=False,
                   dictcols_return=True, drop_index=False):

    df_concat = pd.DataFrame()

    for ix, element in enumerate(tuple_of_joins):

        if ix == 0:
            index = element[1]
            df_concat = dict_df[element[0]][0]
            continue

        df = dict_df[element[0]][0]

        # Drop Duplicated Index
        df = df.loc[~df.index.duplicated(keep='first')]

        try:
            df.reset_index(inplace=True, drop=drop_index)

        except Exception:
            print('The index of the table ' + element[0] +
                  ' is already a column of the table. Please, \
                  switch drop_index to True')

        cols = [col for col in df.columns if col not in df_concat.columns]
        cols = list(dict.fromkeys(cols + element[2]))

        df_concat = df_concat.merge(df[cols],
                                    left_on=element[1],
                                    right_on=element[2],
                                    how=element[3],
                                    validate=element[4])

    df_concat.set_index(index, inplace=True)

    if df_2pickle:
        obj_to_pickle(df_concat, df_2pickle, 'L0_Input_Dataframe.pkl')

        if dictcols_return:
            dict_cols = ddicts2dict_of_columns(dict_df)
            obj_to_pickle(dict_cols, df_2pickle, 'L0_DictofColumns.pkl')

    if df_return:

        if dictcols_return:
            return (df_concat, ddicts2dict_of_columns(dict_df))

        return df_concat


"""
 <<< pandas_profiling(df, pathname)>>>

 Producing Exploratory Data Analysis in Pandas Profile Html Report

 Parameters
 _ _ _ _ _

 df : DataFrame

 pathname : String
    Path + Report name with .html extension

 Returns
 _ _ _ _ _

 Pandas Profile Html report file
"""


def pandas_profiling(df, pathname):

    profile = pp.ProfileReport(df)
    profile.to_file(pathname)

    print('Pd_Profiling_Report' + ' exported in ' + pathname)
