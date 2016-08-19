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
    daily_return.columns = ['daily return']
    weekly_return = data_df.pct_change(periods=5).to_frame()
    weekly_return.columns = ['weekly return']
    monthly_return = data_df.pct_change(periods=22).to_frame()
    monthly_return.columns = ['monthly return']
    three_month_return = data_df.pct_change(periods=66).to_frame()
    three_month_return.columns = ['three month return']
    six_month_return = data_df.pct_change(periods=126).to_frame()
    six_month_return.columns = ['six month return']
    
    return daily_return, weekly_return , monthly_return, three_month_return, six_month_return

def linear_line(data_df):
    
    print(data_df.iloc[:,[0,]])
 
    line_df = np.polyfit(data_df.iloc[:,[0,]].values, data_df.iloc[:,[1,]], 1)
    
    #print(line_df)
'''  
    r_x, r_y = zip(*((i, i*line_df[0] + line_df[1]) for i in data_df.length))
    line_df = pd.DataFrame({
    'length' : r_x,
    'weight' : r_y
        })
        
    return line_df
'''
def retail_sales_test(data_df):
    

    release_date = data_df['Retail Sales Release'].dropna()
    daily_return, weekly_return, monthly_return, three_month_return, six_month_return = spx_return(data_df['S&P 500 Price'])
    weekly_return = weekly_return.shift(-5)
    monthly_return = monthly_return.shift(-22)
    release_date = release_date.map(lambda x: str(x)[:-2]).values
    release_date = set(pd.to_datetime(release_date).tolist())
    
    diff_df = (data_df['Retail Sales Exp MoM'] - data_df['Retail Sales MoM']).dropna().to_frame()
    diff_df.columns = ['diff']

    diff_df = diff_df.reindex(release_date)
    
    diff_df.dropna(inplace=True)

    daily_test = pd.concat([diff_df,daily_return,six_month_return],axis=1).dropna(axis=0)
    
    daily_test_conditional_equity_up = daily_test[daily_test['six month return'] > 0]
    daily_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    daily_test_conditional_equity_down = daily_test[daily_test['six month return'] <= 0]
    daily_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
  

    ax = daily_test_conditional_equity_up.plot(kind='scatter', x='diff', y='daily return', color='Red', label='Up Market')
    
    daily_test_conditional_equity_down.plot(kind='scatter', x='diff', y='daily return',color='Grey', label='Down Market', ax=ax)
    
    
    weekly_test = pd.concat([diff_df,weekly_return,six_month_return],axis=1).dropna(axis=0)
    
    
    
    
    
    weekly_test_conditional_equity_up = weekly_test[weekly_test['six month return'] > 0]
    weekly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    weekly_test_conditional_equity_down = weekly_test[weekly_test['six month return'] <= 0]
    weekly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = weekly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='weekly return', color='Red', label='Up Market')
    
    weekly_test_conditional_equity_down.plot(kind='scatter', x='diff', y='weekly return',color='Grey', label='Down Market', ax=ax)
        

    monthly_test = pd.concat([diff_df,monthly_return,six_month_return],axis=1).dropna(axis=0)

    monthly_test_conditional_equity_up = monthly_test[monthly_test['six month return'] > 0]
    monthly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    monthly_test_conditional_equity_down = monthly_test[monthly_test['six month return'] <= 0]
    monthly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = monthly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='monthly return', color='Red', label='Up Market')
    
    monthly_test_conditional_equity_down.plot(kind='scatter', x='diff', y='monthly return',color='Grey', label='Down Market', ax=ax)   
 
def ism_manufacturing_test(data_df):
    

    release_date = data_df['ISM Manufacturing Release'].dropna()
    daily_return, weekly_return, monthly_return, three_month_return, six_month_return = spx_return(data_df['S&P 500 Price'])
    weekly_return = weekly_return.shift(-5)
    monthly_return = monthly_return.shift(-22)
    release_date = release_date.map(lambda x: str(x)[:-2]).values
    release_date = set(pd.to_datetime(release_date).tolist())
    
    diff_df = (data_df['ISM Manufacturing Exp'] - data_df['ISM Manufacturing']).dropna().to_frame()
    diff_df.columns = ['diff']

    diff_df = diff_df.reindex(release_date)
    
    diff_df.dropna(inplace=True)

    daily_test = pd.concat([diff_df,daily_return,six_month_return],axis=1).dropna(axis=0)
    
    daily_test_conditional_equity_up = daily_test[daily_test['six month return'] > 0]
    daily_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    daily_test_conditional_equity_down = daily_test[daily_test['six month return'] <= 0]
    daily_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = daily_test_conditional_equity_up.plot(kind='scatter', x='diff', y='daily return', color='Red', label='Up Market')
    
    daily_test_conditional_equity_down.plot(kind='scatter', x='diff', y='daily return',color='Grey', label='Down Market', ax=ax)
    
    
    weekly_test = pd.concat([diff_df,weekly_return,six_month_return],axis=1).dropna(axis=0)
    
    
    
    
    
    weekly_test_conditional_equity_up = weekly_test[weekly_test['six month return'] > 0]
    weekly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    weekly_test_conditional_equity_down = weekly_test[weekly_test['six month return'] <= 0]
    weekly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = weekly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='weekly return', color='Red', label='Up Market')
    
    weekly_test_conditional_equity_down.plot(kind='scatter', x='diff', y='weekly return',color='Grey', label='Down Market', ax=ax)
        

    monthly_test = pd.concat([diff_df,monthly_return,six_month_return],axis=1).dropna(axis=0)

    monthly_test_conditional_equity_up = monthly_test[monthly_test['six month return'] > 0]
    monthly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    monthly_test_conditional_equity_down = monthly_test[monthly_test['six month return'] <= 0]
    monthly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = monthly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='monthly return', color='Red', label='Up Market')
    
    monthly_test_conditional_equity_down.plot(kind='scatter', x='diff', y='monthly return',color='Grey', label='Down Market', ax=ax)   
    
def payroll_test(data_df):
    

    release_date = data_df['Payrolls Release'].dropna()
    daily_return, weekly_return, monthly_return, three_month_return, six_month_return = spx_return(data_df['S&P 500 Price'])
    weekly_return = weekly_return.shift(-5)
    monthly_return = monthly_return.shift(-22)
    release_date = release_date.map(lambda x: str(x)[:-2]).values
    release_date = set(pd.to_datetime(release_date).tolist())
    
    diff_df = (data_df['Payrolls Exp'] - data_df['Payrolls']).dropna().to_frame()
    diff_df.columns = ['diff']

    diff_df = diff_df.reindex(release_date)
    
    diff_df.dropna(inplace=True)

    daily_test = pd.concat([diff_df,daily_return,six_month_return],axis=1).dropna(axis=0)
    
    daily_test_conditional_equity_up = daily_test[daily_test['six month return'] > 0]
    daily_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    daily_test_conditional_equity_down = daily_test[daily_test['six month return'] <= 0]
    daily_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = daily_test_conditional_equity_up.plot(kind='scatter', x='diff', y='daily return', color='Red', label='Up Market')
    
    daily_test_conditional_equity_down.plot(kind='scatter', x='diff', y='daily return',color='Grey', label='Down Market', ax=ax)
    
    
    weekly_test = pd.concat([diff_df,weekly_return,six_month_return],axis=1).dropna(axis=0)   
    
    
    weekly_test_conditional_equity_up = weekly_test[weekly_test['six month return'] > 0]
    weekly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    weekly_test_conditional_equity_down = weekly_test[weekly_test['six month return'] <= 0]
    weekly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = weekly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='weekly return', color='Red', label='Up Market')
    
    weekly_test_conditional_equity_down.plot(kind='scatter', x='diff', y='weekly return',color='Grey', label='Down Market', ax=ax)
        

    monthly_test = pd.concat([diff_df,monthly_return,six_month_return],axis=1).dropna(axis=0)

    monthly_test_conditional_equity_up = monthly_test[monthly_test['six month return'] > 0]
    monthly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    monthly_test_conditional_equity_down = monthly_test[monthly_test['six month return'] <= 0]
    monthly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = monthly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='monthly return', color='Red', label='Up Market')
    
    monthly_test_conditional_equity_down.plot(kind='scatter', x='diff', y='monthly return',color='Grey', label='Down Market', ax=ax)   


def citi_surprise_test(data_df):
    
    data_df = data_df.reindex(pd.date_range(data_df.index[0],data_df.index[-1],freq='m'),method='ffill')
    equity_monthly_rtn = data_df['S&P 500 Price'].pct_change(periods=1).to_frame().dropna()
    absolute_monthly_test = pd.concat([data_df['Citi Suprise'],equity_monthly_rtn],axis=1).dropna(axis=0)
    absolute_monthly_test.columns = ['Citi Suprise','S&P 500 Monthly Return']
    citi_above_zero_test = absolute_monthly_test[absolute_monthly_test['Citi Suprise'] > 0]
    
    citi_below_zero_test = absolute_monthly_test[absolute_monthly_test['Citi Suprise'] <= 0]
  
    
    ax = citi_above_zero_test.plot(kind='scatter', x='Citi Suprise', y='S&P 500 Monthly Return', color='Red', label='Citi Surprise > 0')

    citi_below_zero_test.plot(kind='scatter', x='Citi Suprise', y='S&P 500 Monthly Return',color='Grey', label='Citi Surprise <= 0', ax=ax)   


    citi_three_month_average = pd.rolling_mean(data_df['Citi Suprise'],window=3)
    citi_six_month_average = pd.rolling_mean(data_df['Citi Suprise'],window=12)
    citi_trend = pd.DataFrame(citi_three_month_average - citi_six_month_average,index=citi_six_month_average.index)
    
    citi_trend_test = pd.concat([citi_trend,equity_monthly_rtn],axis=1).dropna(axis=0)
    citi_trend_test.columns = ['Citi Suprise','S&P 500 Monthly Return']
    
    
    
    citi_up_trend_test = citi_trend_test[citi_trend_test['Citi Suprise'] > 0]
    
    citi_down_trend_test = citi_trend_test[citi_trend_test['Citi Suprise'] <= 0]
  
    
    ax = citi_up_trend_test.plot(kind='scatter', x='Citi Suprise', y='S&P 500 Monthly Return', color='Red', label='Citi Surprise Up Trend')

    citi_down_trend_test.plot(kind='scatter', x='Citi Suprise', y='S&P 500 Monthly Return',color='Grey', label='Citi Surprise Down Trend', ax=ax)   

    
    
    
    
    #citi_trend_up_test = absolute_monthly_test[absolute_monthly_test['Citi Suprise'] > 0]
    
    #citi_trend_down_test = absolute_monthly_test[absolute_monthly_test['Citi Suprise'] <= 0]


'''
    release_date = data_df['Payrolls Release'].dropna()
    daily_return, weekly_return, monthly_return, three_month_return, six_month_return = spx_return(data_df['S&P 500 Price'])
    weekly_return = weekly_return.shift(-5)
    monthly_return = monthly_return.shift(-22)
    release_date = release_date.map(lambda x: str(x)[:-2]).values
    release_date = set(pd.to_datetime(release_date).tolist())
    
    diff_df = (data_df['Payrolls Exp'] - data_df['Payrolls']).dropna().to_frame()
    diff_df.columns = ['diff']

    diff_df = diff_df.reindex(release_date)
    
    diff_df.dropna(inplace=True)

    daily_test = pd.concat([diff_df,daily_return,six_month_return],axis=1).dropna(axis=0)
    
    daily_test_conditional_equity_up = daily_test[daily_test['six month return'] > 0]
    daily_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    daily_test_conditional_equity_down = daily_test[daily_test['six month return'] <= 0]
    daily_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = daily_test_conditional_equity_up.plot(kind='scatter', x='diff', y='daily return', color='Red', label='Up Market')
    
    daily_test_conditional_equity_down.plot(kind='scatter', x='diff', y='daily return',color='Grey', label='Down Market', ax=ax)
    
    
    weekly_test = pd.concat([diff_df,weekly_return,six_month_return],axis=1).dropna(axis=0)   
    
    
    weekly_test_conditional_equity_up = weekly_test[weekly_test['six month return'] > 0]
    weekly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    weekly_test_conditional_equity_down = weekly_test[weekly_test['six month return'] <= 0]
    weekly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = weekly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='weekly return', color='Red', label='Up Market')
    
    weekly_test_conditional_equity_down.plot(kind='scatter', x='diff', y='weekly return',color='Grey', label='Down Market', ax=ax)
        

    monthly_test = pd.concat([diff_df,monthly_return,six_month_return],axis=1).dropna(axis=0)

    monthly_test_conditional_equity_up = monthly_test[monthly_test['six month return'] > 0]
    monthly_test_conditional_equity_up.drop(['six month return'],axis=1,inplace=True)
    
    monthly_test_conditional_equity_down = monthly_test[monthly_test['six month return'] <= 0]
    monthly_test_conditional_equity_down.drop(['six month return'],axis=1,inplace=True)
    
    ax = monthly_test_conditional_equity_up.plot(kind='scatter', x='diff', y='monthly return', color='Red', label='Up Market')
    
    monthly_test_conditional_equity_down.plot(kind='scatter', x='diff', y=
'''


   
    
#retail_sales_testtickers = ['Retail Sales Exp MoM','Retail Sales MoM','Retail Sales Release','S&P 500 Price']
#retail_sales_return_df = retail_sales_test(ticker_data[retail_sales_test_tickers])

#ism_man_test_tickers = ['ISM Manufacturing Exp','ISM Manufacturing','ISM Manufacturing Release','S&P 500 Price']
#ism_man_return_df = ism_manufacturing_test(ticker_data[ism_man_test_tickers])

#payrolls_test_tickers = ['Payrolls Exp','Payrolls','Payrolls Release','S&P 500 Price']
#payrolls_return_df = payroll_test(ticker_data[payrolls_test_tickers])


citi_surprise_tickers = ['Citi Suprise','S&P 500 Price']
citi_surprise_return_df = citi_surprise_test(ticker_data[citi_surprise_tickers])
