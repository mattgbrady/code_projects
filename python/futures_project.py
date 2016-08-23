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


merged_data = quandl.get(['SCF/CME_CL1_OB','SCF/CME_CL2_OB'])

print(merged_data)
