# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:14:17 2016

@author: ubuntu
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

mpl.rcParams['font.size'] = 6
plt.rcParams.update({'axes.titlesize': 'small'})


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)


def get_recession_periods(data_df):
    data_df['scores'] =  np.where(data_df < 0, -1, 0)
    temp_df = pd.DataFrame()
    
    data_df['rolling_sum_2'] = pd.rolling_sum(data_df['scores'],window=2)
    #temp_df['rolling_sum_2'] = temp_df['rolling_sum_2'].shift(-1)

    #temp_df['rolling_sum_2'].fillna(0,inplace=True)

    
    data_df['recession_flag'] = data_df['rolling_sum_2'] + data_df['scores'] 
    pass


def scenario(ticker,NBER_rec, data_df):
    
    combined_NBER_factor = data_df.join(NBER_rec).fillna(method='ffill')
    
    
    recession_factor_df = combined_NBER_factor[combined_NBER_factor['USRECD']==1].dropna(axis=0, how='any')
    expansion_factor_df = combined_NBER_factor[combined_NBER_factor['USRECD']==0].dropna(axis=0, how='any')
    
    return recession_factor_df[ticker], expansion_factor_df[ticker]
    
    pass
#start date
dt_start = pd.datetime(1900,1,1)

ticker_dict = {'GDP CQOQ Index': 'US GDP','USURTOT Index': 'Unemployment','NAPMPMI Index': 'ISM Manufacturing',
               'CPI YOY Index': 'CPI Headline', 'NFP TCH Index': 'Payrolls', 'CONSSENT Index': 'U of M Confidence',
               'RSTAYOY Index': 'US Retail Sales'}
               
ticker_list = list(ticker_dict.keys())    
               

bl_data = bbg.get_histData(ticker_list,['PX_LAST'],dt_start,pd.datetime.today().date())

bl_data = bl_data.pivot(index='date', columns='Ticker', values='PX_LAST')

NBER_rec = pdr.DataReader(['USRECD'],data_source='fred',start=dt_start)
NBER_rec.index.name='date'

for ticker in ticker_list:
    
    if ticker != 'USURTOT Index':
        continue
    
    NBER_rec_temp = NBER_rec.copy()
 
    temp_tick_data = bl_data[ticker].to_frame()
    temp_tick_data.dropna(inplace=True)
    
    look_back_sma = 24
    hl_ewm = 6
    ticker_ma = (temp_tick_data-pd.rolling_mean(temp_tick_data,window=look_back_sma))/pd.rolling_std(temp_tick_data,window=look_back_sma)
    ticker_z_ema = (temp_tick_data-pd.ewma(temp_tick_data,halflife=hl_ewm))/pd.ewmstd(temp_tick_data,halflife=hl_ewm)
    
    
    recession_level, expansion_level  = scenario(ticker,NBER_rec_temp, temp_tick_data)
    recession_ma, expansion_ma  = scenario(ticker,NBER_rec_temp, ticker_ma)
    recession_ema, expansion_ema  = scenario(ticker,NBER_rec_temp, ticker_z_ema)
    
 
    #plot
 
    
    new_idx = pd.date_range(temp_tick_data.index.to_datetime()[0],pd.datetime.today().date(),freq='B')
    temp_tick_data = temp_tick_data.reindex(new_idx,method='ffill')
    
    NBER_rec_temp = NBER_rec_temp.reindex(new_idx,method='ffill')
    

    plt_dts = new_idx.to_pydatetime()
    plt_dts = mdates.date2num(plt_dts)
    
    fig,ax = plt.subplots(2,2)
    ax1 = ax[0,0]    
    ax2 = ax[0,1]    
    ax3 = ax[1,0]    
    ax4 = ax[1,1]
    
    ax1.plot(plt_dts,temp_tick_data.values,'k-', color='g')
    ax1.set_title('Raw')
    ymin,ymax = ax1.get_ylim()
   
    ax1.set_ylim([ymin,ymax])
    
    ax1.fill_between(plt_dts,ymin,ymax,where=NBER_rec_temp.values.T[0]>0,facecolor='grey',alpha=0.4)
    ax1.set_xlim([pd.to_datetime(temp_tick_data.index.values[0]),pd.to_datetime(temp_tick_data.index.values[-1])])

    ax1.legend([ticker_dict[ticker]],fontsize=6)

    
    

    recession_level.plot(kind='kde',ax=ax2,color='b', title='Level')
    expansion_level.plot(kind='kde',ax=ax2,color='r')     
    
    
    level_test = stats.ks_2samp(recession_level, expansion_level)
    
    
    recession_ma.plot(kind='kde',ax=ax3,color='b',title=str(look_back_sma) +' M Rolling Z')
    expansion_ma.plot(kind='kde',ax=ax3,color='r')   
    
    ma_test = stats.ks_2samp(recession_ma, expansion_ma)
    
    
    recession_ema.plot(kind='kde',ax=ax4,color='b', title=str(hl_ewm) +' M half-life Z')
    expansion_ema.plot(kind='kde',ax=ax4,color='r')
    
    
    ema_test = stats.ks_2samp(recession_ema, expansion_ema)

   
    
    ax2.legend(['recession','non-recession'], fontsize=6)

    ax3.legend(['recession','non-recession'],fontsize=6)
    ax4.legend(['recession','non-recession'],fontsize=6)
    
  
    #fig.show()
    
    
    fig.savefig('pdf/'+ticker_dict[ticker]+'.pdf')
    plt.close(fig)

    
    
    

