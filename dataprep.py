import pandas as pd
import numpy as np
import pickle

def col_numeric_cat_split(df: pd.DataFrame) -> list:
    """
    takes in a pandas dataframe
    returns a list of numeric columns, and another list of categorical columns
    """
    num_cols = df._get_numeric_data().columns
    cols = df.columns
    categorical_cols = list(set(cols) - set(num_cols))
    num_cols = list(num_cols)
    return num_cols, categorical_cols


class dummies():
    def __init__(self):
        self.columns_dict = {}

    def fit(self, df: pd.DataFrame, cols: list, nan_dummies=True, prefix=None, prefix_sep="-"):
        """
        fits the dummies object to the training data

        keyword arguments:
        df -- the DataFrame of training data
        cols -- list of columns to create dummies for

        optional arguments:
        nan_dummies -- Include a column for nan values, defualt True
        prefix -- "cols": prepends the original DataFrame column name to the dummy column names
                  None: Does not include a prefix
                  deafult: None
        prefix_sep -- string: what is used to seperate the prefix and the dummy name.
                      default: "-"
        """
        for col in cols:
            dummy_cols = list(set(df[col].unique()) - set([np.nan]))
            if prefix == "cols":
                named_dummies = []
                for dummy in dummy_cols:
                    named_dummies.append(f"{col}{prefix_sep}{dummy}")
                self.columns_dict[col] = named_dummies
            else:
                self.columns_dict = dummy_cols
            
            if nan_dummies == True:
                self.columns_dict[col].append(f"{col}{prefix_sep}nan")


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates dummy columns for the DataFrame provided. Creates the columns from the fit DataFrame,
        removes any columns it creates dummies for.

        keyword argument:
        df -- DataFrame to create dummy columns for

        returns:
        df -- DataFrame containing the dummy columns
        """
        for k,v in self.columns_dict.items():
            for cat in v:
                df[cat] = (df[k] == cat) * 1
            df = df.drop(k, axis=1)
        return df

    def save_pickle(self, dest_filename: str) -> None:
        """ store the fit data in a pickle object for later use """
        with open(dest_filename, "wb") as f:
            pickle.dump(self.columns_dict, f)

    def load_pickle(self, dest_filename: str) -> None:
        """ loads a previous stored pickle object of fit data """
        with open(dest_filename, "r") as f:
            self.columns_dict = pickle.load(f)
