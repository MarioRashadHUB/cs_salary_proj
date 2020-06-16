# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv('cs_data_cleaned.csv')

df['avg_salary'] = df.apply(lambda x: x.avg_salary*1000, axis =1)

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue',
             'job_state','same_state','age','python_yn','java_yn','C_plus_plus_yn',
             'C_sharp_yn','PHP_yn','swift_yn','ruby_yn','javascript_yn','SQL_yn','job_simp','seniority']]

# get dummy data to convert categorical variable into dummy/indicator variables
df_dum = pd.get_dummies(df_model)
df_dum

df_real = df[["avg_salary", "python_yn","java_yn","C_plus_plus_yn",
             "C_sharp_yn","PHP_yn","swift_yn","ruby_yn","javascript_yn","SQL_yn"]]


"""# Train and Test Split (80/20)"""
from sklearn.model_selection import train_test_split

X = df_real.drop('avg_salary', axis = 1)
X
y = df_real.avg_salary
y.value_counts()

# 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Model Building (Random Forrest)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

rf = RandomForestRegressor(max_depth=2, random_state=0)

# Tuning Random Forest using GridSearch
from sklearn.model_selection import GridSearchCV

params = {'n_estimators':range(10,100,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs_rf = GridSearchCV(rf, params ,scoring = 'neg_mean_absolute_error', cv = 10)
gs_rf.fit(X_train, y_train)

gs_rf.best_score_
best_model = gs_rf.best_estimator_




import pickle

# save the model to disk
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)


#pickle.dump(xgb_best, open('model.pkl','wb'))
#pickle.dump(lm, open('model.pkl','wb'))

# load the model to compare the results
#model = pickle.load(open('model.pkl','rb'))




