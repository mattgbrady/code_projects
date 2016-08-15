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
def get_data(ticker_list):
    
    dt_start = pd.datetime(1999,1,1)

    bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date(),freq='MONTHLY')
    
    
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

ticker_data = pd.DataFrame()


#download data one time, aftewards import csv file
'''
for ticker in new_ticker_list:
    ticker_list = [ticker]
    monthly_returns_df = get_data(ticker_list)
    
    ticker_data = ticker_data.append(monthly_returns_df, ignore_index=True)
 '''   

#ticker_data.to_csv('spx_member_prices.csv')
spx_ticker_df = get_data(['SPXT INDEX'])
spx_ticker_df.to_csv('spx_total_prices.cvs')
'''
ticker_data = pd.read_csv('spx_member_prices.csv', index_col=0)


ticker_data.drop_duplicates(subset=['date','Ticker'],inplace=True)

ticker_data = ticker_data.pivot(index='date', columns='Ticker', values='PX_LAST')
ticker_data.index = pd.to_datetime(ticker_data.index)


ticker_data = ticker_data.reindex(pd.date_range(ticker_data.index[0],ticker_data.index[-1],freq='M'),method='ffill')
ticker_data = ticker_data.reindex(data_df.columns.values)

monthly_returns_df = ticker_data.pct_change(periods=1)

#convert to same type

data_df.columns = pd.to_datetime(data_df.columns)
number_of_stocks = 50


portfolio_monthly_rt_df = pd.DataFrame()

for runs in range(1,5):
    monthly_return = []
    print(runs)
    for date in monthly_returns_df.index:
        loop_df = data_df[date]

        loop_df.dropna(inplace=True)
    
        loop_ticker_list = loop_df.values
    
        new_loop_ticker_list = []
    
    
        for word in loop_ticker_list:
            new_word = word[:-2] + 'US Equity'
            new_loop_ticker_list.append(new_word)
    
        #choose random 50 stocks
        random_stock_list = []
        counter=0
        while len(random_stock_list) < number_of_stocks:
            ticker = random.choice(new_loop_ticker_list)
            #remove stock from old list in loop
            new_loop_ticker_list.remove(ticker)
            random_stock_list.extend([ticker])
            counter = counter + 1
    
        #query monthly_returns_df each month using random list and dataframe
        loop_returns_df = monthly_returns_df.loc[date].to_frame().dropna()
        loop_returns_df = pd.DataFrame(loop_returns_df)

        loop_returns_df = loop_returns_df.ix[random_stock_list]
        loop_returns_df['weights'] = 1/number_of_stocks
        loop_returns_df['weight_return'] = loop_returns_df[date] * loop_returns_df['weights']
    
        loop_returns_df['weight_return']

        monthly_return.append(loop_returns_df['weight_return'].sum())
        
    temp_return_df = pd.DataFrame(monthly_return,index=monthly_returns_df.index,columns=['test '+str(runs)])
    portfolio_monthly_rt_df = pd.concat([portfolio_monthly_rt_df,temp_return_df],axis=1)
print(portfolio_monthly_rt_df)
    #print(loop_returns_df.columns.values)
    #print(loop_returns_df.index.values)
    #print(random_stock_list)
    #print(type(loop_returns_df))
    #print(loop_returns_df.index)
    #loop_returns_df.set_index(random_stock_list)
    
    #print(loop_returns_df)
'''


