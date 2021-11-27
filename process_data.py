#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# load messages dataset
messages = pd.read_csv('disaster_messages.csv')
messages.head()


# In[ ]:


# load categories dataset
categories = pd.read_csv('disaster_categories.csv')
categories.head()


# In[ ]:


# merge datasets
df = messages.merge(categories, on = ['id'])
df.head()


# In[ ]:


# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand = True)
categories.head()


# In[ ]:


# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.transform(lambda x: x[:-2]).tolist()
print(category_colnames)


# In[ ]:


# rename the columns of `categories`
categories.columns = category_colnames
categories.head()


# In[ ]:


# Convert category values to just numbers 0 or 1
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].transform(lambda x: x[-1:])
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
categories.head()


# In[ ]:


# drop the original categories column from `df`
df.drop('categories', axis = 1, inplace = True)

df.head()


# In[ ]:


# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis = 1)
df.head()


# In[ ]:


# check number of duplicates
sum(df.duplicated())


# In[ ]:


# drop duplicates
df.drop_duplicates(inplace = True)


# In[ ]:


# check number of duplicates
sum(df.duplicated())


# In[ ]:


engine = create_engine('sqlite:///Database.db')
df.to_sql('DatabaseTable', engine, index=False)

