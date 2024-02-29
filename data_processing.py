import os
import pandas as pd
import numpy as np

from zipfile import ZipFile
import pyspark
import matplotlib.pyplot as plt
import math
import time
import re
import zipfile
import warnings
from tqdm import tqdm
from utils import *

# Ignore DeprecationWarning
warnings.filterwarnings("ignore")
#warnings.resetwarnings()


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

f_name = file.infolist()[0].filename
f_name#[:f_name.find('/')]

atts = ['advertiser_id', 'platform', 'latitude', 'longitude',
       'horizontal_accuracy', 'location_at','speed']


targets=[i for i in os.listdir() if i[-3:]=='zip']
sum2 = dict()
for target in targets:
    zip_file = ZipFile(target)
    file_list = zip_file.infolist()
    print("Working in "+target+'...')
    i=1
    for file in tqdm(file_list):
        df2 = pd.read_csv(zip_file.open(file.filename),usecols=['advertiser_id','platform'])
        df_grouped2 = df2.groupby('advertiser_id').count()
        sum2[file] = df_grouped2.to_dict()

sum2_ori = {i:sum2[i]['platform'] for i in sum2}
test = [list(sum2_ori[i].keys()) for i in sum2_ori]
test = [item for sublist in test for item in sublist]
sum2 = pd.DataFrame(sum2_ori)
sum2.to_csv('sum2.csv')

sum3 = sum2.apply(lambda x:x.isna().sum(),axis=1)
sum3.to_csv('sum3_new.csv')


remain = sum3[sum3<=160]


user_data = []
for target in targets:
    zip_file = ZipFile(target)
    file_list = zip_file.infolist()
    print("Working in "+target+'...')
    i=1
    for file in tqdm(file_list):
        df2 = pd.read_csv(zip_file.open(file.filename))
        user_data.append(df2[df2['advertiser_id'].isin(select),usecols=atts])        
        

files = os.listdir()
data = []
i=0
for file in tqdm(files):
    data.append(pd.read_csv(file,index_col='advertiser_id').iloc[:,-45:])

df = pd.concat(data)
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
df_np = df.groupby('advertiser_id').apply(lambda x:x.to_numpy())
user_names = df_np.index
user_data = np.array([i for i in df_np])


with open('../user_name.txt','w') as f:
    f.write('\n'.join(user_names))

np.save('../user_data.npy',user_data)



