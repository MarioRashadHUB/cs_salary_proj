#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 22:25:25 2020

@author: mtss
"""


import glassdoor_scraper as gs
import pandas as pd
path = "/Users/mtss/Documents/cs_salary_proj/chromedriver"

# Scrapes 20000 job listings for each title from glassdoor and saves them to dataframes
df = gs.get_jobs('software engineer', 20000, False, path, 15)


# Stores all dataframes into .CSV files
df_brand_amb.to_csv('glassdoor_jobs',index = False)
