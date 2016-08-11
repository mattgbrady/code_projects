# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:39:10 2016

@author: Matt Brady
"""

import pandas as pd
import bbgREST as bbg
from pandas_datareader import data as pdr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.dates as mdates
import matplotlib
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as mpl
import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp
import bt as bt

import plotly.plotly as py
from plotly.tools import FigureFactory as FF


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

           


#data from bloomberg
def get_data(ticker_list):

    dt_start = pd.datetime(1999,1,1)

    bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date())
    
    
    return bl_data


data = pd.read_excel('spx_index_members.xlsx', index_col=0)
data_df = data.copy()
data_df = data_df.reset_index()
date_index = data_df.columns.values

#get unique tickers in S&P 500 between dates
ticker_list = []
for date in date_index:
    ticker_loop_df = data_df[date]
    ticker_loop_df.dropna(inplace=True)
    
    ticker_loop_list = ticker_loop_df.values

    ticker_list.extend(ticker_loop_list)
    
unique_list = list(set(ticker_list))
new_ticker_list = []
for word in unique_list:
    new_word = word[:-2] + 'US Equity'
    new_ticker_list.append(new_word)


counter = 2
#len(new_ticker_list)

ticker_data = pd.DataFrame()

for ticker in new_ticker_list:
    print(ticker)
    monthly_returns_df = get_data(ticker)
    ticker_data.append(monthly_return_df)


print(ticker_data)

