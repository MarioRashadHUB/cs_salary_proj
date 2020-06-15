#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:28:48 2020

@author: MARIO
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv('cs_data_cleaned.csv')

df['avg_salary'] = df.apply(lambda x: x.avg_salary*1000, axis =1)

# choose revelant columns
df.columns


df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue',
             'job_state','same_state','age','python_yn','java_yn','C_plus_plus_yn',
             'C_sharp_yn','PHP_yn','swift_yn','ruby_yn','javascript_yn','SQL_yn','job_simp','seniority']]


# get dummy data / creates categorical data into integers
df_dum = pd.get_dummies(df_model)

# train test split (80/20)
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values # creates an array - recommended to use for models

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 1)

# multiple linear regressions

# single linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

lm = LinearRegression()

# train the model
lm.fit(X_train, y_train)

# preform prediction on the test data
y_pred = lm.predict(X_test)

# performance metrics
print('Coefficients:', lm.coef_)
print('Intercept: ', lm.intercept_)
print('Mean absolute error (MAE): %.2f' % mean_absolute_error(y_test, y_pred))


# 10 Fold Cross Validation (to generalize data)
from sklearn.model_selection import cross_val_score

np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))

# lasso regression  (L1 Regularization)
from sklearn.linear_model import Lasso

lm_las = Lasso() # alpha defaults to 1
lm_las.fit(X_train, y_train)
np.mean(cross_val_score(lm_las, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))

# Find the optimal alpha

alpha = []
err = []

for i in range(1, 100):
  alpha.append(i/100)
  lmlas = Lasso(alpha = (i/100))
  err.append(np.mean(cross_val_score(lmlas, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 10)))

plt.plot(alpha,err)

# shows that alpha is best at 0.99 (shows where optimal alpha is)
err = tuple(zip(alpha,err))
df_err = pd.DataFrame(err, columns = ['alpha','err'])
df_err[df_err.err == max(df_err.err)]

lm_las = Lasso(0.99) # alpha defaults to 1
lm_las.fit(X_train, y_train)
np.mean(cross_val_score(lm_las, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))

# Ridge Regression (L2 Regularization)
from sklearn.linear_model import Ridge

lm_rid = Ridge()
lm_rid.fit(X_train, y_train)
np.mean(cross_val_score(lm_rid, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 10))

alpha = []
err = []

for i in range(1, 100):
  alpha.append(i/100)
  lmrid = Ridge(alpha = (i/100))
  err.append(np.mean(cross_val_score(lmrid, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 10)))

plt.plot(alpha,err)

# Random Forrest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=0)

np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))

# Tuning Random Forest using GridSearch
from sklearn.model_selection import GridSearchCV

params = {'n_estimators':range(10,100,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs_rf = GridSearchCV(rf, params ,scoring = 'neg_mean_absolute_error', cv = 10)
gs_rf.fit(X_train, y_train)

gs_rf.best_score_
gs_rf.best_estimator_

# XGBoost ()
from xgboost import XGBClassifier

# initialize the linear regression model
xgb = XGBClassifier()

# train the model
xgb.fit(X_train, y_train)

np.mean(cross_val_score(xgb ,X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 10))

# Tuning XGBoost using GridSearchCV
params = {'min_child_weight': [3, 5], 'gamma': [0.5, 1], 'subsample': [0.8, 1.0],
          'colsample_bytree': [0.6, 0.8], 'max_depth': [1,2]}

gs_xgb = GridSearchCV(xgb, params ,scoring = 'neg_mean_absolute_error', cv = 10)
gs_xgb.fit(X_train, y_train)

gs_xgb.best_score_

gs_xgb.best_estimator_

# Comparing model performances
lm_pred = lm.predict(X_test)
lm_las_pred = lm_las.predict(X_test)
lm_rid_pred = lm_rid.predict(X_test)
rf_pred = gs_rf.best_estimator_.predict(X_test)
xgb_pred = gs_xgb.best_estimator_.predict(X_test)

print("MLR MAE: ", mean_absolute_error(y_test, lm_pred))
print("Lasso Regression MAE: ", mean_absolute_error(y_test, lm_las_pred))
print("Ridge Regression MAE: ", mean_absolute_error(y_test, lm_rid_pred))
print("Random Forest MAE: ", mean_absolute_error(y_test, rf_pred))
print("XGBoost MAE: ", mean_absolute_error(y_test, xgb_pred))

"""
MLR MAE:  5866922120737.591
Lasso Regression MAE:  19647.73318939012
Ridge Regression MAE:  19014.24458024136
Random Forest MAE:  18593.04930865262
XGBoost MAE:  22670.52023121387
"""
