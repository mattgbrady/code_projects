# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:26:17 2016

@author: ubuntu
"""

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
def get_data(ticker_dict):

    ticker_list = list(ticker_dict.keys())    

    dt_start = pd.datetime(1900,1,1)

    bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date())

    bl_data = bl_data.pivot(index='date', columns='Ticker', values='PX_LAST')
    
    return bl_data
    



ticker_dict = {'USGG10YR Index': 'US 10yr','SPX Index': 'S&P 500'}
               

data_df = get_data(ticker_dict)

ewm_df = pd.ewma(data_df, min_periods=252,halflife=252)


new_df = data_df - ewm_df

us_ten = new_df['USGG10YR Index'].dropna()
spx_ten = new_df['SPX Index'].dropna()


print('here')