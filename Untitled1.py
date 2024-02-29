#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
os.chdir(r'D:\dy\prj')
from zipfile import ZipFile
import pyspark
import matplotlib.pyplot as plt
import math
import time
import re
import zipfile
import warnings

# Ignore DeprecationWarning
warnings.filterwarnings("ignore")
#warnings.resetwarnings()


# In[2]:


def regenerate(zip_file,zip2):
    # regenerate as in standard form
    #step1: categorize
    def gendict(filelist):
        day_pattern = r'\bday=(\d+)\b'
        adict = dict()
        for f in filelist:
            day = re.search(day_pattern, f).group(1)
            if day in adict.keys():
                adict[day].append(f)
            else:
                adict[day] = [f]
        return adict
    #step2: read and concatenate files in list
    def concat(sub_list):
        df = pd.DataFrame()
        for filename in sub_list:
            new_df = pd.read_csv(zip_file.open(filename))
            df = pd.concat([df,new_df],ignore_index=True)
        return df 
    #step3: save and compress
    def save_comp(df,name,month,zip2):
        df.to_csv(name,index=False)
        outname = month+'/'+name
        with ZipFile(zip2,'w') as zip_ref:
            zip_ref.write(name,outname,compress_type = zipfile.ZIP_DEFLATED)
        os.remove(name)
        
    filelist = [f for f in zip_file.namelist() if f[-3:]=='csv']
    filedict = gendict(filelist)
    temp = filelist[0]
    MONTH = temp[:temp.find('/')]
    month_pattern = r'\bmonth=(\d+)\b'
    month = re.search(month_pattern,temp).group(1)
    for day in filedict.keys():
        df = concat(filedict[day])
        name = 'Bonston_'+MONTH+day+'_2019.csv'
        save_comp(df,name,month,zip2)  
        print('day'+day+' finish')
    #pull out


# In[39]:


dict
pattern = r'\bday=(\d+)\b'
matches = re.findall(pattern, f_name)


# In[86]:


f_name = file.infolist()[0].filename
f_name#[:f_name.find('/')]


# In[4]:


atts = ['advertiser_id', 'platform', 'latitude', 'longitude',
       'horizontal_accuracy', 'location_at','speed']


# In[2]:


targets=[i for i in os.listdir() if i[-3:]=='zip']
targets


# In[3]:


targets=[i for i in os.listdir() if i[-3:]=='zip']
sum2 = dict()
for target in targets:
    zip_file = ZipFile(target)
    file_list = zip_file.infolist()
    print("Working in "+target+'...')
    i=1
    for file in file_list:
        df2 = pd.read_csv(zip_file.open(file.filename),usecols=['advertiser_id','platform'])
        df_grouped2 = df2.groupby('advertiser_id').count()
        sum2[file] = df_grouped2.to_dict()
        print(f"-----------------finishing {i}/{len(file_list)}------------------")
        i += 1


# In[5]:


sum2_ori = {i:sum2[i]['platform'] for i in sum2}
test = [list(sum2_ori[i].keys()) for i in sum2_ori]
test = [item for sublist in test for item in sublist]
sum2 = pd.DataFrame(sum2_ori)


# In[12]:


sum2.to_csv('sum2.csv')


# In[6]:


sum3 = sum2.apply(lambda x:x.isna().sum(),axis=1)


# In[8]:


sum3.to_csv('sum3_new.csv')


# In[11]:


remain = sum3[sum3<=160]
remain


# In[9]:


sum3


# In[ ]:


sep


# In[53]:


import random
select = []
for i in range(100):
    select.append(random.choice(remain.index))


# In[58]:


user_data = []
for target in targets:
    zip_file = ZipFile(target)
    file_list = zip_file.infolist()
    print("Working in "+target+'...')
    i=1
    for file in file_list:
        df2 = pd.read_csv(zip_file.open(file.filename))
        user_data.append(df2[df2['advertiser_id'].isin(select),usecols=atts])        
        print(f"-----------------finishing {i}/{len(file_list)}------------------")
        i += 1


# In[3]:


import plotly.express as px
import datetime


# In[4]:


df_all = pd.concat(user_data)


# In[5]:


df_all = pd.read_csv('100user.csv',index_col=0)


# In[6]:


df_all['time']=pd.to_datetime(df_all['location_at'], unit='s', utc=True).dt.tz_convert('US/Eastern')
df_all['hour'] = df_all['time'].apply(lambda x:x.hour)
df_all['date'] = df_all['time'].apply(lambda x:x.date)
df_all['month'] = df_all['time'].apply(lambda x:x.month)
df_all['weekday'] = df_all['time'].apply(lambda x:x.weekday)


# In[7]:


select =df_all['advertiser_id'].unique()


# In[8]:


grouped = df_all.groupby('advertiser_id')
user_all = {name:grouped.get_group(name) for name in select}


# In[ ]:





# In[13]:


newpath = r'D:\dy\prj\user_pics' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
for i in range(100):
    fig = px.scatter_mapbox(user_all[select[i]], lat = 'latitude', lon = 'longitude',color='speed',size_max=0.001,zoom=11)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
    fig.update_layout(width=1000, height=600)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(showlegend=False)
    fig.write_image(r".\user_pics\user"+str(i)+".png") 


# In[14]:


newpath = r'D:\dy\prj\user_pics_byhour' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
for i in range(100):
    fig = px.scatter_mapbox(user_all[select[i]], lat = 'latitude', lon = 'longitude',color='hour',size_max=0.001,zoom=11)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
    fig.update_layout(width=1000, height=600)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(showlegend=False)
    fig.write_image(r".\user_pics_byhour\user"+str(i)+".png") 


# In[15]:


newpath = r'D:\dy\prj\user_pics_byweekday' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
for i in range(100):
    fig = px.scatter_mapbox(user_all[select[i]], lat = 'latitude', lon = 'longitude',color='weekday',size_max=0.001,zoom=11)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
    fig.update_layout(width=1000, height=600)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(showlegend=False)
    fig.write_image(r".\user_pics_byweekday\user"+str(i)+".png") 


# In[11]:


user = user_all[select[5]]
fig = px.scatter_mapbox(user[user['speed']<1],lat = 'latitude', lon = 'longitude',size_max=0.001,zoom=11)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
fig.update_layout(width=1000, height=600)
fig.update_layout(coloraxis_showscale=False)
fig.update_layout(showlegend=False)
fig.show()


# In[99]:


for j in range(100):
    print('user'+str(j)+'+'*10)
    user = user_all[select[j]]
    months = sorted(user['month'].unique())
    num_month = len(user['month'].unique())
    fig,ax = plt.subplots(math.ceil(num_month/2)+2,2,figsize=(6,10))
    user['weekday'].hist(ax=ax[0][0],color='r',bins=7)
    user['hour'].hist(ax=ax[0][1],color='r')
    user[user['weekday']>4]['hour'].hist(ax=ax[1][0],color='g')
    user[user['weekday']<=4]['hour'].hist(ax=ax[1][1],color='g')
    i=0
    for month in months:
        user[user['month']==month]['hour'].hist(ax=ax[2+i//2][i%2])
        i+=1
        print(month)
    plt.show()
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


dt = pd.read_csv('sum3_new.csv',index_col=0)


# In[32]:


dt = dt.apply(lambda s:214-s)


# In[50]:


y,x = np.histogram(dt,bins=np.arange(0,220,20))

plt.bar(x[1:],y,width=20)
plt.axvline(143,color='red',linestyle='--')
plt.grid(True,alpha=0.3)


# In[54]:


dt.apply(lambda x:x>142).sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:


def get_sum(text_file):
    df = pd.read_csv(zip_file.open(text_file.filename),usecols=['advertiser_id','platform'])
    df_grouped = df.groupby('advertiser_id').count()
    return df_grouped.to_dict()


# In[111]:


conf = pyspark.SparkConf().setAppName('infostop_phase_1').set('spark.driver.memory',4).set('spark.executor.memory',32).set('spark.worker.memory',32).setMaster('local[*]')
sc = pyspark.SparkContext().getOrCreate(conf)

#sc.parallelize(zip_file.infolist()).mapPartitions(get_sum).collect()


# In[ ]:




