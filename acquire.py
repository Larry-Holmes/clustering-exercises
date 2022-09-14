import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import os
from env import host, username, password


def get_connection(db, user=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT 
                *
                From predictions_2017 
                JOIN properties_2017 USING (parcelid)
                LEFT JOIN propertylandusetype USING (propertylandusetypeid)
                WHERE transactiondate Like "2017%%"
                AND propertylandusedesc IN('Single Family Residential','Inferred Single Family Residential');
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df


