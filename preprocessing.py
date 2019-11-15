import numpy as np
import pandas as pd
import gc
import project_preprocessing_helper
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


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

    return df

def split_train_test_target(df):
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

### Bureau Balance: extract min max balance length of balance, count each dummy variable of status ###
def read_bureau_balance():
    previous=pd.read_csv('./home-credit-default-risk/bureau_balance.csv')
    agg_balance=previous.drop(['STATUS'],axis=1).groupby('SK_ID_BUREAU').agg([min,max,'count'])
    agg_balance.columns=['bureau_balance_min','bureau_balance_max','bureau_balance_count']
    agg_status=pd.get_dummies(previous.drop(['MONTHS_BALANCE'],axis=1)).groupby('SK_ID_BUREAU').agg('mean')
    del previous
    gc.collect()
    agg_status.columns=['STATUS_0', 'STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C', 'STATUS_X']
    agg_balance=agg_balance.merge(agg_status,on='SK_ID_BUREAU',how='left')
    return agg_balance


def read_bureau():
    bureau = pd.read_csv('./home-credit-default-risk/bureau.csv')
    bureau = bureau.drop(columns=['AMT_ANNUITY', 'AMT_CREDIT_MAX_OVERDUE'])
    bureau_balance = read_bureau_balance()
    #bureau_balance = bureau_balance.drop(['STATUS'], axis=1)
    bureau = bureau.merge(bureau_balance, right_index=True, left_on='SK_ID_BUREAU', how='left')
    bureau = project_preprocessing_helper.df_agg(bureau, 'SK_ID_CURR', 'bureau')
    return bureau

### AGG min max for numeric columns, AGG counts for factor columns ###
def read_POS_balance():
    previous=pd.read_csv('./home-credit-default-risk/POS_CASH_balance.csv')
    previous = previous.fillna(0)
    previous = previous.drop('SK_ID_PREV', axis=1)
    factor = [col for col in previous.columns if previous[col].dtype == 'object']
    pre_num = previous.drop(factor, axis=1)
    factor.append('SK_ID_CURR')
    pre_factor = previous[factor]
    agg_num = pre_num.groupby('SK_ID_CURR').agg(['min', 'max','count'])
    col_name = []
    for i in pre_num.columns[1:]:
        for j in ['min', 'max','count']:
            col_name.append('{}_{}'.format(i, j))
    agg_num.columns = col_name
    agg_factor = pd.get_dummies(pre_factor, dummy_na=True).groupby('SK_ID_CURR').agg('mean')
    del previous
    gc.collect()
    agg_num=agg_num.merge(agg_factor,on='SK_ID_CURR',how='left')
    return agg_num

### Extract min max median from installation file ###
def read_Install_balance():
    previous=pd.read_csv('./home-credit-default-risk/installments_payments.csv')
    previous = previous.fillna(0)
    previous = previous.drop('SK_ID_PREV', axis=1)
    agg = previous.groupby('SK_ID_CURR').agg(['min', 'max', 'median'])
    col_name = []
    for i in previous.columns[1:]:
        for j in ['min', 'max', 'median']:
            col_name.append('{}_{}'.format(i, j))
    agg.columns = col_name
    return agg

### Extract min max median from Card Balance and mean the dummy variables ###
def read_Card_balance():
    previous = pd.read_csv('./home-credit-default-risk/credit_card_balance.csv')
    previous = previous.drop('SK_ID_PREV', axis=1)
    agg_factor = pd.get_dummies(previous[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']]).groupby('SK_ID_CURR').agg('mean')
    agg = previous.drop('NAME_CONTRACT_STATUS', axis=1).groupby('SK_ID_CURR').agg(['min', 'max', 'mean'])
    col_name = []
    for i in previous.drop('NAME_CONTRACT_STATUS', axis=1).columns[1:]:
        for j in ['min', 'max', 'mean']:
            col_name.append('{}_{}'.format(i, j))
    agg.columns = col_name
    agg = agg.merge(agg_factor, on='SK_ID_CURR', how='left')
    return agg

### Get train test data and wirte into .csv ###
def get_train_test_data():
    data = read_bureau()
    pre=read_application()
    train, test, target, features_name=split_train_test_target(pre)
    test.to_csv('test.csv')
    del test,target,features_name,pre
    gc.collect()
    train=train.merge(data,on='SK_ID_CURR',how='left')
    del data
    gc.collect()

    data_merge = read_POS_balance()
    train = train.merge(data_merge, on='SK_ID_CURR', how='left')
    del data_merge
    gc.collect()
    data_merge = read_Install_balance()
    train = train.merge(data_merge, on='SK_ID_CURR', how='left')
    del data_merge
    gc.collect()
    data_merge = read_Card_balance()
    train = train.merge(data_merge, on='SK_ID_CURR', how='left')
    del data_merge
    gc.collect()

    pre=read_previous_application()
    train=train.merge(pre,on='SK_ID_CURR',how='left')
    train.to_csv("train.csv")
    return
