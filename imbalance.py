import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np
from imblearn.ensemble import EasyEnsemble
from sklearn import linear_model
from imblearn.under_sampling import RandomUnderSampler


# EazyEnsemble: Undersampling
def ezensemble(X_train, y_train):
    a = list(X_train)
    ee = EasyEnsemble(random_state=0, n_subsets=10)
    ee.fit(X_train, y_train)
    X_resampled, y_resampled = ee.fit_sample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled[1], columns=a)
    y_resampled = pd.DataFrame(y_resampled[1], columns=['Target'])
    return X_resampled, y_resampled


# SMOTE: Oversampling
def smote(X_train, y_train):
    a = list(X_train)
    smo = SMOTE(random_state=123)
    over_samples_X, over_samples_y = smo.fit_sample(X_train, y_train)
    over_samples_X = pd.DataFrame(over_samples_X, columns=a)
    over_samples_y = pd.DataFrame(over_samples_y, columns=['Target'])
    return over_samples_X, over_samples_y


# RandomUnderSampler: Undersampling
def RUS(X_train, y_train):
    a = list(X_train)
    rus = RandomUnderSampler()
    X_resampled, y_resampled = rus.fit_sample(X_train, y_train)
    X_resampled = pd.DataFrame(X_resampled, columns=a)
    y_resampled = pd.DataFrame(y_resampled, columns=['Target'])
    return X_resampled, y_resampled


# Only for logistic regression
def weightlog(X_train, y_train, X_test):
    logreg = linear_model.LogisticRegression(C=1e5, class_weight={0: .08, 1: .92})
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    return y_pred


# Duplicating data
def dup(data):
    dup_times = int(data['TARGET'].value_counts()[0] / data['TARGET'].value_counts()[1])
    data_1 = data[data['TARGET'] == 1]
    big_data_1 = data_1.copy()
    for i in range(dup_times - 2):
        big_data_1 = pd.concat([big_data_1, data_1])
    data = pd.concat([data, big_data_1])
    return data
