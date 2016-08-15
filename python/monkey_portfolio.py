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
    
    dt_start = pd.datetime(1992,12,1)

    bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date(),freq='MONTHLY')
    
    
    return bl_data
'''

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

for ticker in new_ticker_list:
    ticker_list = [ticker]
    monthly_returns_df = get_data(ticker_list)
    
    ticker_data = ticker_data.append(monthly_returns_df, ignore_index=True)


ticker_data.to_csv('spx_member_prices.csv')

spx_ticker_df = get_data(['SPXT INDEX'])
spx_ticker_df.to_csv('spx_total_prices.cvs')
'''
ticker_data = pd.read_csv('spx_member_prices.csv', index_col=0)
spx_ticker = pd.read_csv('spx_total_prices.cvs', index_col=0)
spx_ticker = spx_ticker.pivot(index='date', columns='Ticker', values='PX_LAST')
spx_ticker.index = pd.to_datetime(spx_ticker.index)


ticker_data.drop_duplicates(subset=['date','Ticker'],inplace=True)

ticker_data = ticker_data.pivot(index='date', columns='Ticker', values='PX_LAST')
ticker_data.index = pd.to_datetime(ticker_data.index)

ticker_data = ticker_data.reindex(pd.date_range(ticker_data.index[0],ticker_data.index[-1],freq='M'),method='ffill')
ticker_data = ticker_data.reindex(data_df.columns.values)

monthly_returns_df = ticker_data.pct_change(periods=1)

#convert to same type

data_df.columns = pd.to_datetime(data_df.columns)
number_of_stocks = 2


portfolio_monthly_rt_df = pd.DataFrame()
portfolio_random_mt_df = pd.DataFrame()

for runs in range(0,2):
    monthly_return = []
    print(runs)
    for date in monthly_returns_df.index:
        random_weights = []
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

        monthly_return.append(loop_returns_df['weight_return'].sum())
        
        for number in range(0,number_of_stocks):
            random_weights.append(random.uniform(0,100))
        sum_weights = sum(random_weights)
        random_weights = [x/sum_weights for x in random_weights]
        random_returns = loop_returns_df[date].values
        new_array = random_weights * random_returns
        random_weight_return = sum(new_array)
    
        print(random_returns)
        
    temp_return_df = pd.DataFrame(monthly_return,index=monthly_returns_df.index,columns=['test '+str(runs)])
    temp_random_df = pd.DataFrame(random_weight_return,index=monthly_returns_df.index,columns=['test '+str(runs)])
    portfolio_random_mt_df = pd.concat([portfolio_random_mt_df,temp_random_df],axis=1)
    portfolio_monthly_rt_df = pd.concat([portfolio_monthly_rt_df,temp_return_df],axis=1)


portfolio_monthly_rt_df = portfolio_monthly_rt_df.add(1).cumprod()
portfolio_random_mt_df = portfolio_random_mt_df.add(1).cumprod()

spx_ticker = spx_ticker.pct_change(periods=1)

spx_ticker = spx_ticker.reindex(pd.date_range(portfolio_monthly_rt_df.index[0],portfolio_monthly_rt_df.index[-1],freq='M'),method='ffill')
spx_ticker = spx_ticker.add(1).cumprod()

portfolio_monthly_rt_df.to_csv('random_50_equal_rebalance.csv')
portfolio_random_mt_df.to_csv('random_50_random_weight_rebalance.csv')
spx_ticker.to_csv('spx_total_return.csv')

ax = plt.subplot()
ax1 = ax
ax2 = ax
ax1.semilogy(portfolio_random_mt_df,color='grey')

ax2.semilogy(spx_ticker)

#portfolio_monthly_rt_df.append(spx_ticker)
plt.show()
#spx_ticker.plot()
'''
random_50_equal_rebalance = pd.read_csv('random_50_equal_rebalance.csv', index_col=0)
spx_total_return = pd.read_csv('spx_total_return.csv', index_col=0)

median_tot_return = random_50_equal_rebalance.ix[max(random_50_equal_rebalance.index.values)].median()

median_value =(abs(random_50_equal_rebalance.ix[max(random_50_equal_rebalance.index.values)] - median_tot_return)).to_frame()
temp_df =  (random_50_equal_rebalance.ix[max(random_50_equal_rebalance.index.values)] + median_value)
column_name = (median_value.sort(columns=[max(random_50_equal_rebalance.index.values)])[:1].index.values[0])

median_equity_curve = random_50_equal_rebalance[column_name]
#statistics of Median
median = random_50_equal_rebalance.pct_change(periods=1).median(axis=1).to_frame()
median = median.add(1).cumprod()
median.columns = ['Median']

combined = pd.concat([median_equity_curve,spx_total_return],axis=1)
combined = pd.concat([median,combined],axis=1)
combined.plot(logy=True)


monthly_diff = pd.DataFrame()

monthly_diff['diff'] = combined['Median'].pct_change(periods=1) - combined['SPXT INDEX'].pct_change(periods=1)



success_rate = len(monthly_diff[monthly_diff['diff'] > 0])


print(success_rate/len(monthly_diff))
'''