#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt
os.chdir(r'D:\dy\prj')
from math import sin, cos, sqrt, atan2, radians, degrees
from sklearn.cluster import DBSCAN
import json
from sklearn.neighbors import NearestNeighbors
import warnings 
warnings.filterwarnings('ignore') 


targets = os.listdir('./user_data1')
done = os.listdir('./user_data')
problematic = ['1000_1100.csv','3600_3700.csv','4100_4200.csv']
targets = [i for i in targets if i not in done]
#targets = [i for i in targets if i not in problematic]


for i in range(len(targets)):
    in_path = './user_data1/'+targets[i]
    print(f'----------{i+1}/{len(targets)}--------')
    print(in_path)
    sample = pd.read_csv(in_path,index_col=0)
    sample = sample_process(sample)
    df = sample.groupby('advertiser_id').apply(lambda user:user_transform(user,True))
    out_path = './user_data/'+targets[i]
    df.to_csv(out_path)
    
    print(out_path)

in_path = './user_data1/'+targets[0]
sample = pd.read_csv(in_path,index_col=0)
sample = sample_process(sample)

users = np.unique(sample['advertiser_id'])
problems = ['94030A83-26C2-4F02-B6D6-898C769D6AA8','84B42011-9717-4855-803F-5043A984D447','245D3C52-1917-4743-982A-06BCA2E27203','2825C708-7307-4D1E-ACCC-082D8EA273FA']
users = [i for i in users if i not in problems]
dfs = [user_transform(sample[sample['advertiser_id']==user],verbose=True) for user in users]

resample = pd.concat([simple_func(dfs[i],users[i]) for i in range(len(users))])

out_path = './user_data/'+targets[0]
resample.to_csv(out_path)

import time 
s = time.time()
df = sample.groupby('advertiser_id').apply(user_transform)
e = time.time()
print(e-s)






