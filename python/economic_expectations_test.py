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
import random

import plotly.plotly as py
from plotly.tools import FigureFactory as FF


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)         

#data from bloomberg
def get_data(ticker_list,fld):
    
    ticker = [ticker_list[0]]
    
    dt_start = pd.datetime(1992,12,1)

    bl_data = bbg.get_histData(ticker,[fld],dt_start,pd.datetime.today().date(),freq='DAILY')

    bl_data = bl_data.pivot(index='date', columns='Ticker', values=fld)
    
    bl_data.columns = [ticker_list[1]]
    return bl_data


'''
#get unique tickers in S&P 500 between dates
retail_sales_expectation = get_data(['RSTAMOM Index','Retail Sales Exp MoM'],'BN SURVEY MEDIAN')
retail_sales_mom = get_data(['RSTAMOM Index','Retail Sales MoM'],'PX_LAST')
retail_sales_release = get_data(['RSTAMOM Index','Retail Sales Release'],'ECO_RELEASE_DT')
citi_surprise = get_data(['CESIUSD Index','Citi Suprise'],'PX_LAST')
payrolls = get_data(['NFP TCH Index','Payrolls'],'PX_LAST')
payrolls_expectations = get_data(['NFP TCH Index','Payrolls Exp'],'BN SURVEY MEDIAN')
payrolls_release = get_data(['NFP TCH Index','Payrolls Release'],'ECO_RELEASE_DT')
ism_man = get_data(['NAPMPMI Index','ISM Manufacturing'],'PX_LAST')
ism_man_expectations = get_data(['NAPMPMI Index','ISM Manufacturing Exp'],'BN SURVEY MEDIAN')
ism_man_release = get_data(['NAPMPMI Index','ISM Manufacturing Release'],'ECO_RELEASE_DT')




spx_total_return = get_data(['SPXT Index','S&P 500 TR'],'PX_LAST')
spx_return = get_data(['SPX Index','S&P 500 Price'],'PX_LAST')
data_array = [retail_sales_expectation,retail_sales_mom,retail_sales_release,payrolls,citi_surprise,
              payrolls,payrolls_expectations,payrolls_release,ism_man,ism_man_expectations,ism_man_release,
              spx_total_return,spx_return]





ticker_data = pd.DataFrame()



for ticker_df in data_array:

    ticker_df = ticker_df.reindex(pd.date_range(ticker_df.index[0],ticker_df.index[-1],freq='B'),method='ffill')
    ticker_data = pd.concat([ticker_data, ticker_df], axis=1)
   

ticker_data.fillna(method='ffill',inplace=True)

ticker_data.to_csv('ticker_data.csv')
'''
ticker_data = pd.read_csv('ticker_data.csv', index_col=0)

ticker_data.index = pd.to_datetime(ticker_data.index)

def spx_return(data_df):
    
    daily_return = data_df.pct_change(periods=1).to_frame()
    weekly_return = data_df.pct_change(periods=5).to_frame()
    monthly_return = data_df.pct_change(periods=22).to_frame()
    
    return daily_return, weekly_return , monthly_return

def retail_sales_test(data_df):
    

    release_date = data_df['Retail Sales Release'].dropna()
    daily_return, weekly_return, monthly_return = spx_return(data_df['S&P 500 Price'])
    weekly_return = weekly_return.shift(-5)
    monthly_return = monthly_return.shift(-22)
    release_date = release_date.map(lambda x: str(x)[:-2]).values
    release_date = set(pd.to_datetime(release_date).tolist())
    
    diff_df = (data_df['Retail Sales Exp MoM'] - data_df['Retail Sales MoM']).dropna().to_frame()
    diff_df.columns = ['diff']

    diff_df = diff_df.reindex(release_date)
    
    diff_df.dropna(inplace=True)

    daily_test = pd.concat([diff_df,daily_return],axis=1).dropna(axis=0)
    
    daily_test.plot(kind='scatter', title='Retail Sales vs Daily Return',x='diff', y='S&P 500 Price')

    weekly_test = pd.concat([diff_df,weekly_return],axis=1).dropna(axis=0)
    
    weekly_test.plot(kind='scatter',title='Retail Sales vs Weekly Return',  x='diff', y='S&P 500 Price')
    
    monthly_test = pd.concat([diff_df,monthly_return],axis=1).dropna(axis=0)
    
    monthly_test.plot(kind='scatter',title='Retail Sales vs Monthly Return', x='diff', y='S&P 500 Price')

def ism_manufacturing_test(data_df):
    

    release_date = data_df['ISM Manufacturing Release'].dropna()
    daily_return, weekly_return, monthly_return = spx_return(data_df['S&P 500 Price'])
    weekly_return = weekly_return.shift(-5)
    monthly_return = monthly_return.shift(-22)
    release_date = release_date.map(lambda x: str(x)[:-2]).values
    release_date = set(pd.to_datetime(release_date).tolist())
    
    diff_df = (data_df['ISM Manufacturing Exp'] - data_df['ISM Manufacturing']).dropna().to_frame()
    diff_df.columns = ['diff']

    diff_df = diff_df.reindex(release_date)
    
    diff_df.dropna(inplace=True)

    daily_test = pd.concat([diff_df,daily_return],axis=1).dropna(axis=0)
    
    daily_test.plot(kind='scatter',title='ISM Manufacturing vs Daily Return', x='diff', y='S&P 500 Price')

    weekly_test = pd.concat([diff_df,weekly_return],axis=1).dropna(axis=0)
    
    weekly_test.plot(kind='scatter', title='ISM Manufacturing vs Weekly Return',x='diff', y='S&P 500 Price')
    
    monthly_test = pd.concat([diff_df,monthly_return],axis=1).dropna(axis=0)
    
    monthly_test.plot(kind='scatter',title='ISM Manufacturing vs Monthly Return', x='diff', y='S&P 500 Price')
    
def payroll_test(data_df):
    

    release_date = data_df['Payrolls Release'].dropna()
    daily_return, weekly_return, monthly_return = spx_return(data_df['S&P 500 Price'])
    weekly_return = weekly_return.shift(-5)
    monthly_return = monthly_return.shift(-22)
    release_date = release_date.map(lambda x: str(x)[:-2]).values
    release_date = set(pd.to_datetime(release_date).tolist())
    
    diff_df = (data_df['Payrolls Exp'] - data_df['Payrolls']).dropna().to_frame()
    diff_df.columns = ['diff']

    diff_df = diff_df.reindex(release_date)
    
    diff_df.dropna(inplace=True)

    daily_test = pd.concat([diff_df,daily_return],axis=1).dropna(axis=0)
    
    daily_test.plot(kind='scatter',title='Payrolls vs Daily Return', x='diff', y='S&P 500 Price')

    weekly_test = pd.concat([diff_df,weekly_return],axis=1).dropna(axis=0)
    
    weekly_test.plot(kind='scatter', title='Payrolls vs Weekly Return',x='diff', y='S&P 500 Price')
    
    monthly_test = pd.concat([diff_df,monthly_return],axis=1).dropna(axis=0)
    
    monthly_test.plot(kind='scatter',title='Payrolls vs Monthly Return', x='diff', y='S&P 500 Price')
    
    
retail_sales_test_tickers = ['Retail Sales Exp MoM','Retail Sales MoM','Retail Sales Release','S&P 500 Price']
retail_sales_return_df = retail_sales_test(ticker_data[retail_sales_test_tickers])

ism_man_test_tickers = ['ISM Manufacturing Exp','ISM Manufacturing','ISM Manufacturing Release','S&P 500 Price']
ism_man_return_df = ism_manufacturing_test(ticker_data[ism_man_test_tickers])

payrolls_test_tickers = ['Payrolls Exp','Payrolls','Payrolls Release','S&P 500 Price']
payrolls_return_df = payroll_test(ticker_data[payrolls_test_tickers])