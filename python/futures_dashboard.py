# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:23:22 2016

@author: Matt Brady
"""
import pandas as pd
import quandl
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from datetime import datetime

import pandas.io.data as web



py.sign_in('mattgbrady','cmjs1osucd')
quandl.ApiConfig.api_key = 'YMqrB1SXyjSUkTHYwpJ2'



'''
ticker_list = ['SCF/CME_CL1_FW','SCF/CME_CL1_FR','SCF/CME_CL1_FB']

merged_data = quandl.get(ticker_list)



raw_data = merged_data[['SCF/CME_CL1_FW - Settle','SCF/CME_CL1_FR - Settle','SCF/CME_CL1_FB - Settle']]

raw_data.to_csv('raw_data.csv')
'''
data_df = pd.read_csv('raw_data.csv',index_col=0)

data_df_shift = data_df.shift(252)

data_df_diff = data_df - data_df_shift
data_df_diff.dropna(inplace=True)



data_format = [
    go.Scatter(
        x=data_df_diff.index, # assign x as the dataframe column 'x'
        y=data_df_diff['SCF/CME_CL1_FW - Settle'],
        name='Calendar Weighted'
    ),
    go.Scatter(
        x=data_df_diff.index, # assign x as the dataframe column 'x'
        y=data_df_diff['SCF/CME_CL1_FR - Settle'],
        name='Backwards Ratio'
    ),
    go.Scatter(
        x=data_df_diff.index, # assign x as the dataframe column 'x'
        y=data_df_diff['SCF/CME_CL1_FB - Settle'],
        name='Backwards Panama'
    ),

]



layout = dict(
    title='Time series with range slider and selectors',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(step='all'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(count=5,
                    label='5yr',
                    step='year',
                    stepmode='backward'),
                dict(count=10,
                    label='10yr',
                    step='year',
                    stepmode='backward')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data_format, layout=layout)
py.iplot(fig, filename='Crude', validate=False, layout=layout)



data_format = [
    go.Scatter(
        x=data_df.index, # assign x as the dataframe column 'x'
        y=data_df['SCF/CME_CL1_FW - Settle'],
        name='Calendar Weighted'
    ),
    go.Scatter(
        x=data_df.index, # assign x as the dataframe column 'x'
        y=data_df['SCF/CME_CL1_FR - Settle'],
        name='Backwards Ratio'
    ),
    go.Scatter(
        x=data_df.index, # assign x as the dataframe column 'x'
        y=data_df['SCF/CME_CL1_FB - Settle'],
        name='Backwards Panama'
    )]
fig = dict(data=data_format, layout=layout)
py.iplot(fig, filename='Crude Price', validate=False, layout=layout)

'''
table_df = pd.DataFrame()

table_df = data_df[['SCF/CME_CL1_OB - Settle','one_year']].dropna()

table_df['Percentage'] = (data_df['one_year'] / data_df['SCF/CME_CL1_OB - Settle']) - 1


table_df = table_df[-252:]

table_df.columns = ['Crude CL1','Trailing One Year Crude CL1', 'Percentage Difference']

table = FF.create_table(table_df,index=True, index_title='Date')
py.iplot(table, filename='crude_table')
'''