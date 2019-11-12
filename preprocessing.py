import numpy as np
import pandas as pd
import gc
import project_preprocessing_helper
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


def read_application():
    train = pd.read_csv("./home-credit-default-risk/application_train.csv")
    test = pd.read_csv("./home-credit-default-risk/application_test.csv")

    df = train.append(test)
    del train, test
    gc.collect()

    df = df[df['CODE_GENDER'] != 'XNA']
    df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']
    df = df[df['NAME_INCOME_TYPE'] != 'Maternity leave']

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_EMPLOYED_ABNORMAL'] = (df['DAYS_EMPLOYED'] == np.nan)

    df['RATIO_DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['RATIO_INCOME_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['RATIO_INCOME_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['RATIO_ANNUITY_INCOME'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['RATIO_ANNUITY_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    df = pd.get_dummies(df)

    train = df[df['TARGET'].isnull() == False]
    test = df[df['TARGET'].isnull()]
    target = train['TARGET']
    train = train.drop(columns=['TARGET'])
    test = test.drop(columns=['TARGET'])
    features_name = test.columns

    return train, test, target, features_name


def read_previous_application():
    previous = pd.read_csv("./home-credit-default-risk/previous_application.csv")

    previous = previous.drop(columns=['DAYS_FIRST_DRAWING'])

    previous['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    previous['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    previous['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    previous['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    previous = previous.drop(columns=['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
                              'AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT'])

    previous = project_preprocessing_helper.df_agg(previous, 'SK_ID_CURR', 'previous_app')

    return previous

### extract min max balance length from BUREAU Balance ###
def read_bureau_balance():
    previous=pd.read_csv('./home-credit-default-risk/bureau_balance.csv')
    previous=previous.drop(['STATUS'],axis=1)
    agg=previous.groupby('SK_ID_BUREAU').agg([min,max,'count'])
    agg.columns=['bureau_balance_min','bureau_balance_max','bureau_balance_count']
    return agg
