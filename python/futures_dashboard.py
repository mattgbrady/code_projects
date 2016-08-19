# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:23:22 2016

@author: Matt Brady
"""
import pandas as pd
import quandl
import plotly.plotly as py
import plotly.graph_objs as go



quandl.ApiConfig.api_key = 'YMqrB1SXyjSUkTHYwpJ2'



merged_data = quandl.get(['SCF/CME_CL1_OB','SCF/CME_CL2_OB','SCF/CME_CL3_OB'])

data = merged_data[['SCF/CME_CL1_OB - Settle', 'SCF/CME_CL2_OB - Settle','SCF/CME_CL3_OB - Settle']]


data_format = [
    go.Scatter(
        x=data.index, # assign x as the dataframe column 'x'
        y=data['SCF/CME_CL1_OB - Settle'],
        name='CL1'
    ),
    go.Scatter(
        x=data.index, # assign x as the dataframe column 'x'
        y=data['SCF/CME_CL2_OB - Settle'],
        name='CL2'
    ),
        go.Scatter(
        x=data.index, # assign x as the dataframe column 'x'
        y=data['SCF/CME_CL3_OB - Settle'],
        name='CL3')
]

py.iplot(data_format, filename='crude', validate=False)