#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime


# In[3]:


# Função para converter uma data em texto de um Tweet em um objeto datetime
def str_to_datetime(str_date):
    return datetime.datetime.strptime(str_date, '%a %b %d %H:%M:%S +0000 %Y')

