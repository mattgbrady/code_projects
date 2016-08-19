# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 00:16:50 2016

@author: Matt Brady
"""
import pandas as pd
import quandl
import matplotlib.pyplot as plt
quandl.ApiConfig.api_key = 'YMqrB1SXyjSUkTHYwpJ2'

#y = quandl.bulkdownload("SCF",download_type="complete",filename='../futures.zip')


data = pd.read_csv('SCF_20160819.csv', index_col=0)

print(data)