#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:28:48 2020

@author: MARIO
"""


import pandas as pd
import numpy as np

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

# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

# test ensembles 
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)

#good for classification
mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])
