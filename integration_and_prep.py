#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import pandas as pd
import numpy as np
from tqdm import tqdm

os.chdir(r'D:\dy\prj\user_data')


# In[3]:


def normalize(series):
    return (series-series.min())/(series.max()-series.min())

def winsorize(num,n):
    if num > n:
        num = n
    return num


# In[2]:


files = os.listdir()
data = []
i=0
for file in tqdm(files):
    data.append(pd.read_csv(file,index_col='advertiser_id').iloc[:,-45:])


# In[4]:


df = pd.concat(data)


# In[5]:


df['poi'] = df['poi'].apply(lambda x:winsorize(x,1))
df['poi.1'] = df['poi.1'].apply(lambda x:winsorize(x,1))

dur = df['end_hour']-df['start_hour']
dur = dur.apply(lambda x: x+24 if x<0 else x)
dur = normalize(dur)
speed = normalize(df['speed'])
weekend = df['start_weekend']

start = df['start_hour']
starts = pd.get_dummies(start).iloc[:,1:]

df = pd.concat([df.iloc[:,:-5],speed,dur,weekend,starts],axis=1)


# In[20]:


df.iloc[:,-23:]


# In[17]:


20+20+23+3


# In[10]:


len([i for i in df.columns if '.1' in str(i)])


# In[18]:


df.head().shape


# In[14]:


df_np = df.groupby('advertiser_id').apply(lambda x:x.to_numpy())


# In[15]:


user_names = df_np.index


# In[16]:


user_data = np.array([i for i in df_np])


# In[17]:


with open('../user_name.txt','w') as f:
    f.write('\n'.join(user_names))


# In[18]:


np.save('../user_data.npy',user_data)

