import pandas as pd
import numpy as np
import pickle

def col_numeric_cat_split(df):
    '''
    takes in a pandas dataframe
    returns a list of numeric columns, and another list of categorical columns
    '''
    num_cols = df._get_numeric_data().columns
    cols = df.columns
    categorical_cols = list(set(cols) - set(num_cols))
    num_cols = list(num_cols)
    return num_cols, categorical_cols


class dummies():
    def __init__(self):
        self.columns_dict = {}

    def fit(self, df, cols, nan_dummies=True):
        '''
        stores dictionary of columns and correspomding list of potential values
        '''
        for col in cols:
            self.columns_dict[col] = list(set(df[col].unique()) - set([np.nan]))
            if nan_dummies == True:
                self.columns_dict[col].append('nan_' + col)

    def transform(self, df):
        '''
        takes in a dataframe
        return a transformed dataframe
        '''
        for k,v in self.columns_dict.items():
            for cat in v:
                df[cat] = (df[k] == cat) * 1
            df = df.drop(k, axis=1)
        return df

    def save_pickle(self,destfilename):
        with open(destfilename, "wb") as f:
            pickle.dump(self.columns_dict,f)
    def load_pickle(self,filename):
        with open(filename,"r") as f:
            self.columns_dict = pickle.load(f)
