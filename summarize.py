import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'nums_cols_missing': num_missing, 'percent_cols_missing' : prnt_miss}).\
    reset_index().groupby(['nums_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns ={'customer_id':'count'})
    
    return rows_missing

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
        {
            'num_rows_missing':num_missing,
            'percent_rows_missing':percnt_miss
        })
    return cols_missing

def summarize(df):
    print('DataFrame head: \n')
    print(df.head())
    print('---------')
    print('DataFrame info: \n')
    print(df.info())
    print('---------')
    print('Dataframe Description: \n')
    print(df.describe())
    print('---------')
    print('Null values assessments: \n')
    print('nulls by column: ', nulls_by_col(df))
    print('nulls by row: ', nulls_by_row(df))
    numerical_cols = [col for col in df.columns if df[col].dtype != 'O']
    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    print('---------')
    print('value_counts: \n')
    for col in df.columns:
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins =10, sort= False))
        print('--')
    print('---------')
    print('Report Finished')
    
def dupes(df):
    duplicates_found = []
    no_duplicates = []
    for col in df:
        if df[col].duplicated().sum != '0':
            duplicates_found.append(col)
        else:
            no_duplicates.append(col)
    
    return duplicates_found, no_duplicates