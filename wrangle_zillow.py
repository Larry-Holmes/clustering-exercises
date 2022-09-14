import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

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
                prop.*,
                predictions_2017.logerror,
                predictions_2017.transactiondate,
                air.airconditiongdesc,
                arch.architecturalstyledesc,
                build.buildingclassdesc,
                heat.heatingorsystemdesc,
                landuse.propertylandusedesc,
                story.storydesc,
                construct.typeconstructiondesc
                From properties_2017 prop
                JOIN (
                    Select parcelid, MAX(transactiondate) AS max_transactiondate
                    FROM predictions_2017
                    GROUP BY parcelid
                    ) pred USING(parcelid)
                JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                        AND pred.max_transactiondate = predictions_2017.transactiondate
                LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
                LEFT JOIN storytype story USING (storytypeid)
                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                WHERE prop.latitude IS NOT NULL
                  AND prop.longitude IS NOT NULL
                  AND transactiondate <= '2017-12-31'
                """
                
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def acquire():
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

def nulls_by_rows(df):
    '''
    Get the number and proportion of values per row in the dataframe df
    parameters: single pandas dataframe, df
    return: none
    '''
    return pd.concat([
        df.isna().sum(axis=1).rename('n_missing'),
        df.isna().mean(axis=1).rename('percent_missing'),
    ], axis=1).value_counts().sort_index()


def nulls_by_columns(df):
    '''
    Get the number and proportion of values per column in the dataframe df
    parameters: single pandas dataframe, df
    return: none
    '''
    return pd.concat([
        df.isna().sum().rename('count'),
        df.isna().mean().rename('percent')
    ], axis=1)

def overview(df):
    '''
    print shape of DataFrame, .info() method call, and basic descriptive statistics via .describe()
    parameters: single pandas dataframe, df
    return: none
    '''
    print('--- Shape: {}'.format(df.shape))
    print('--- Info')
    df.info()
    print('--- Column Descriptions')
    print(df.describe(include='all'))

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

def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    
    return df

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
  
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def remove_columns(df, cols_to_remove):  
    df = df.drop(columns=cols_to_remove)

    return df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def split_continuous(df):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    # Take a look at your split datasets

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")
    return train, validate, test

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

def wrangle():
    '''
    acquires, gives summary statistics, and handles missing values contingent on
    the desires of the zillow data we wish to obtain.
    parameters: none
    return: single pandas dataframe, df
    '''
    # grab the data:
    df = acquire()
    # summarize and peek at the data:
    overview(df)
    nulls_by_columns(df).sort_values(by='percent')
    nulls_by_rows(df)
    # task for you to decide: ;)
    # determine what you want to categorize as a single unit property.
    # maybe use df.propertylandusedesc.unique() to get a list, narrow it down with domain knowledge,
    # then pull something like this:
    # df.propertylandusedesc = df.propertylandusedesc.apply(lambda x: x if x in my_list_of_single_unit_types else np.nan)
    # we will drop all missing values for our MVP.
    # In our second iteration, we will tune the proportion and e:
    df = handle_missing_values(df, prop_required_column=.5, prop_required_row=.75)
    # df = df.dropna()
    # take care of any duplicates:
    df = df.drop_duplicates()
    return df