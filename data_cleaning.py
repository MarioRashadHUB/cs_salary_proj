#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:25:47 2020

@author: Mario
"""

import pandas as pd

def title_simplifier(title):
    if 'software engineer' in title.lower():
        return 'software engineer'
    elif 'software developer' in title.lower():
        return 'software engineer'
    elif 'software engineering' in title.lower():
        return 'software engineering'
    elif 'engineer' in title.lower():
        return 'software engineer'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
            return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'


# Imports scrapped dataframe
df = pd.read_csv('software_engineer_jobs')

# decreased dataframe size by 18% by removing all duplicated rows.
df.drop_duplicates(keep=False, inplace=True)

# removes all rows that do not have the State and City listed
df = df[df['Location'] != 'United States']

df['comma'] = df['Location'].apply(lambda x: 1 if ',' in x.lower() else 0)
df = df[df['comma'] != 0]

# simplifies job titles
df['job_simp'] = df['Job Title'].apply(title_simplifier)
df = df[df['job_simp'] != 'na']
df.job_simp.value_counts()

# identifies senior positions
df['seniority'] = df['Job Title'].apply(seniority)
df.seniority.value_counts()


# fix rating and Founding date
df['Rating'] = df['Rating'].clip(lower = 0)


# Salary Parsing

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])

minus_Kd = salary.apply(lambda x: x.replace('K', '').replace('$',''))

min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2


#hourly wage to annual 

df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.hourly ==1 else x.min_salary, axis =1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.hourly ==1 else x.max_salary, axis =1)

# Add's 0's to all min and max salaries 

df['min_salary'] = df.apply(lambda x: x.min_salary*1000, axis =1)
df['max_salary'] = df.apply(lambda x: x.max_salary*1000, axis =1)

# Company name text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

# state field 
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.job_state.value_counts()

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# age of company 
df['age'] = df.Founded.apply(lambda x: x if x <1 else 2020 - x)


# parsing of job description


# Python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df.python_yn.value_counts()

# Java
df['java_yn'] = df['Job Description'].apply(lambda x: 1 if 'java' in x.lower() else 0)
df.java_yn.value_counts()

# C++
df['C_plus_plus_yn'] = df['Job Description'].apply(lambda x: 1 if 'c++' in x.lower() else 0)
df.C_plus_plus_yn.value_counts()

# C#
df['C_sharp_yn'] = df['Job Description'].apply(lambda x: 1 if 'c#' in x.lower() else 0)
df.C_sharp_yn.value_counts()

# PHP
df['PHP_yn'] = df['Job Description'].apply(lambda x: 1 if 'php' in x.lower() else 0)
df.PHP_yn.value_counts()

# Swift
df['swift_yn'] = df['Job Description'].apply(lambda x: 1 if 'swift' in x.lower() else 0)
df.swift_yn.value_counts()

# Ruby
df['ruby_yn'] = df['Job Description'].apply(lambda x: 1 if 'ruby' in x.lower() else 0)
df.ruby_yn.value_counts()

# Javascript
df['javascript_yn'] = df['Job Description'].apply(lambda x: 1 if 'javascript' in x.lower() else 0)
df.javascript_yn.value_counts()

# SQL
df['SQL_yn'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
df.SQL_yn.value_counts()



df.to_csv('cs_data_cleaned.csv',index = False)




