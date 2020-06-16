# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Import data
df = pd.read_csv('cs_data_cleaned.csv')

df['avg_salary'] = df.apply(lambda x: x.avg_salary*1000, axis = 1)
df['senior_yn'] = df['seniority'].apply(lambda x: 1 if 'senior' in x.lower() else 0)

df_model = df[['avg_salary', 'job_state', 'python_yn', 'java_yn', 'C_plus_plus_yn',
             'C_sharp_yn', 'PHP_yn', 'swift_yn', 'ruby_yn', 'javascript_yn', 'SQL_yn',
             'senior_yn']]

labelencoder = LabelEncoder()
df_model['state_code'] = labelencoder.fit_transform(df_model['job_state'])

new = df_model.drop('job_state', axis = 1)

## Model Building (Random Forrest) ##

X = new.drop('avg_salary', axis = 1)
y = new.avg_salary.values

# Train and Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

rf = RandomForestRegressor(max_depth = 2, random_state = 0)
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))

# Tuning Random Forest using GridSearch
params = {'n_estimators':range(10, 100, 10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs_rf = GridSearchCV(rf, params ,scoring = 'neg_mean_absolute_error', cv = 10)
gs_rf.fit(X_train, y_train)

gs_rf.best_score_  # MAE of  -16910.23498264062
best_model = gs_rf.best_estimator_

# Pickle / save model
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)




new = df_model[['job_state', 'state_code']]