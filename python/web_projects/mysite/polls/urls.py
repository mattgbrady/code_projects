# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 00:47:16 2016

@author: ubuntu
"""

from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
]