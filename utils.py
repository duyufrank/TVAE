import os 
import pandas as pd
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
import pyspark
import matplotlib.pyplot as plt
import math
import time
import re
import zipfile
import warnings
from math import sin, cos, sqrt, atan2, radians, degrees
from sklearn.cluster import DBSCAN
import json
from sklearn.neighbors import NearestNeighbors
import warnings 
import logging
import tensorflow as tf
import random

def normalize(series):
    return (series-series.min())/(series.max()-series.min())

def winsorize(num,n):
    if num > n:
        num = n
    return num

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

def get_sum(text_file):
    df = pd.read_csv(zip_file.open(text_file.filename),usecols=['advertiser_id','platform'])
    df_grouped = df.groupby('advertiser_id').count()
    return df_grouped.to_dict()

def user_process(user_data):
    user_data = user_data.sort_values(by='location_at')
    user_data = user_data.set_index('time')
    #user_data = user_data.resample('5T').first()#important
    user_data = user_data.reset_index()
    user_data = user_data.dropna(subset=['latitude'])
    user_data = user_data.reset_index().drop(columns = ['index'])
    return user_data

def sample_process(sample):
    sample['time']=pd.to_datetime(sample['location_at'], unit='s', utc=True).dt.tz_convert('US/Eastern')
    sample['hour'] = sample['time'].apply(lambda x:x.hour)
    sample['date'] = sample['time'].apply(lambda x:x.date)
    sample['month'] = sample['time'].apply(lambda x:x.month)
    sample['weekday'] = sample['time'].apply(lambda x:x.weekday)
    sample['week'] = sample['time'].apply(lambda x:x.week)
    sample['weekend'] = 0
    sample.loc[(sample['hour'] >= 18) & (sample['weekday'] == 4), 'weekend'] = 1
    sample.loc[(sample['hour'] >= 18) & (sample['weekday'] >= 5), 'weekend'] = 1
    sample.loc[(sample['hour'] >= 18) & (sample['weekday'] == 6), 'weekend'] = 0
    return sample

def plot_user(user_data,fill=False):
    if fill:
        user_data.label = user_data.label.fillna(0)
    user_data[['latitude','longitude']].plot.scatter(y='latitude',x='longitude',c=user_data.label,cmap= "plasma",s=0.1)


#identify POI
def user_loc_info(user_data,plot=False,fill=False):
    dbscan = DBSCAN(eps = 10**-3, min_samples = 20).fit(user_data[user_data['speed']<0.5][['latitude','longitude']])
    labels = dbscan.labels_
    num_labels = len(set(labels))
    user_data['label'] = pd.Series(labels,index = user_data[user_data['speed']<0.5].index)
    if num_labels>1:
        a = user_data[(user_data['hour']<7) | (user_data['hour']>=20)].groupby('label').count()    
        temp = user_data.groupby('label').mean()[['latitude','longitude']]
        temp = temp[temp.index>=0]
        if len(a[a.index>=0]['time'])==0:
            idx = np.inf
            home = [99999,99999]
        else:
            idx = a[a.index>=0]['time'].idxmax()#home index
            home = temp[temp.index==idx].squeeze().to_list()
    else:
        home = [99999,99999]
    if num_labels>2:
        POI = temp[temp.index!=idx].T
        POI = [POI.to_dict()[i] for i in POI.to_dict()]
        POI = [list(i.values()) for i in POI]
    else:
        POI = [[99999,99999]]
    if plot:
        plot_user(user_data,fill=fill)
    return(home,POI)

def haversine_distance(lon1, lat1, lon2, lat2):
    # Convert longitude and latitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Radius of the Earth in kilometers
    earth_radius_km = 6371.0

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = earth_radius_km * c

    return distance

def cal_dd(user_data):
    tempdf = pd.concat([user_data[['latitude','longitude']].shift(),user_data[['latitude','longitude']]],axis=1)
    r = tempdf.apply(lambda row: haversine_distance(row[1], row[0], row[3], row[2]),axis=1)
    return r

def select_continous(sequence):
    if len(sequence)<=1:
        return []
    else:
        subseqs = []      
        subseq=[]
        for i in range(len(sequence)):
            if len(subseq)==0:                
                subseq = [sequence[i]]
            else:                
                if sequence[i] == subseq[-1]+1:
                    subseq.append(sequence[i])
                    if i == len(sequence)-1:
                        subseqs.append(subseq)
                else:
                    subseqs.append(subseq)
                    subseq = [sequence[i]]

        return [subseq for subseq in subseqs if len(subseq)>1]

def find_stay1(data):# find the last index of a stay in a continuous time
    a = [[min(i),max(i)] for  i in select_continous(data.index)]
    indexes = [i[1] for i in a]#last one of the period
    times = [data.loc[i[1],'location_at']-data.loc[i[0],'location_at'] for i in a]# corresponding time period of a stay
    return [indexes[i] for i in range(len(indexes)) if times[i] >= (3600*2)]





def find_stay(test):
    dbscan = DBSCAN(eps = 10**-3, min_samples = 5).fit(test[['latitude','longitude']])
    labels = dbscan.labels_
    #test[['latitude','longitude']][:].plot.scatter(y='latitude',x='longitude',c=labels,s=0.5)
    test['labels']=labels
    #test.groupby('labels').apply(find_stay1)

    grouped = test.groupby('labels').apply(find_stay1)
    grouped = grouped[grouped.index>=0]
    grouped = [i for s in grouped for i in s]
    return grouped


def seperater(user_data):
    #natural seperation
    starts = ((user_data['location_at']-user_data['location_at'].shift(1))>3600*2)
    user_data['starts'] = starts
    user_data['ends'] = starts.shift(-1)
    #if they stay but keep recording, look at find_stay
    user_data['action'] = starts.cumsum()
    result = user_data.groupby('action').apply(find_stay)
    stays = [i for s in result for i in s]
    user_data['speed'].loc[stays] = 0 # if stayed, we are pretty sure about the location
    user_data['ends'].loc[stays] = True
    user_data['starts'] = user_data.ends.shift(1).fillna(True)
    user_data['ends'] = user_data['ends'].fillna(True)
    user_data = user_data[user_data['starts']*user_data['ends']!=1]
    user_data['action'] = user_data['starts'].cumsum()
    return user_data


# In[12]:


def user_tracks(user_data_s):
    user_data_s['speed'] = user_data_s['speed'].apply(lambda x:0 if x<0 else x)
    atts = [ 'latitude', 'longitude', 'horizontal_accuracy',  'speed',
           'hour','weekend']
    first = user_data_s.groupby('action')[atts].first()
    last = user_data_s.groupby('action')[atts].last()
    ave = user_data_s.groupby('action')['speed'].mean()
    first.columns = ['start_'+i for i in first.columns]
    last.columns = ['end_'+i for i in last.columns]

    return pd.concat([first,last,ave],axis=1)




def create_uservenue(home,POI):
    att = ['lat','long','types']
    temp = venue_data[att]
    HOME = pd.Series(home+[['home']])
    HOME.index = att
    temp = pd.concat([temp,pd.DataFrame(HOME).T])
    poi = pd.DataFrame(POI)
    poi.columns = ['lat','long']
    poi['types'] = [['poi']]*(poi.shape[0])
    temp = pd.concat([temp,poi])
    temp = temp.reset_index()
    temp = temp.drop('index',axis=1)
    return temp




def locinfo2semantic(loc_info,norm=True):
    adict = {i:0 for i in semantics}
    alist = list(neigh.kneighbors([loc_info[:2]], 1000, return_distance=False).squeeze())

    max_dist = loc_info[2]/1000+np.log(1+loc_info[3])

    candidates = temp.loc[alist]
    candidates['distance'] = candidates.apply(lambda row: haversine_distance(row[1], row[0], loc_info[1], loc_info[0]),axis=1)
    candidates = candidates[candidates['distance']<max_dist]
    # keep one nearest poi
    remove = candidates[candidates['types'].apply(lambda x:x[0]=='poi')].index[1:]
    candidates = candidates[candidates.index.map(lambda x:x not in remove)]
    
    for index in candidates.index:
        distance = candidates.distance[index]
        weight = np.exp(-distance**2)
        sublist = candidates.types[index]
        for i in sublist:
            adict[i]+=weight
    r = pd.Series(adict,index=semantics)
    if norm:
        r[:-2] = r[:-2]/np.sum(r[:-2])
    return r.fillna(0)
    #for item in sublist:
        


#main function-user level
def user_transform(user_data,verbose=False):
    def locinfo2semantic(loc_info,norm=True):
        adict = {i:0 for i in semantics}
        alist = list(neigh.kneighbors([loc_info[:2]], 1000, return_distance=False).squeeze())

        max_dist = loc_info[2]/1000+np.log(1+loc_info[3])

        candidates = temp.loc[alist]
        candidates['distance'] = candidates.apply(lambda row: haversine_distance(row[1], row[0], loc_info[1], loc_info[0]),axis=1)
        candidates = candidates[candidates['distance']<max_dist]
        remove = candidates[candidates['types'].apply(lambda x:x[0]=='poi')].index[1:]
        candidates = candidates[candidates.index.map(lambda x:x not in remove)]
        for index in candidates.index:
            distance = candidates.distance[index]
            weight = np.exp(-distance**2)
            sublist = candidates.types[index]
            for i in sublist:
                adict[i]+=weight
        r = pd.Series(adict,index=semantics)
        if norm:
            r[:-2] = r[:-2]/np.sum(r[:-2])
        return r.fillna(0)
    #reindex
    user_data = user_process(user_data)
    if verbose:
        print(user_data['advertiser_id'].iloc[0])
    home,POI = user_loc_info(user_data)
    #create user-specific venue data, add poi and home to it
    temp = create_uservenue(home,POI)
    #initiate knn to decrease search cost
    neigh = NearestNeighbors(n_neighbors=1000, radius=0.4)
    neigh.fit(temp[['lat','long']])
    #make seperation of start and end. include the seperation of stay
    user_data = seperater(user_data)
    #extract useful information
    user_data = user_tracks(user_data)

    #transfer the location to semantic meaning
    #len(user_track.columns) = 13 6,6,1
    #4,4 need the transformation
    start_s = user_data.iloc[:,:4]
    end_s = user_data.iloc[:,6:10]
    transfered_cols = list(start_s.columns)+ list(end_s.columns)
    remain = [i for i in user_data.columns if i not in transfered_cols]
    remain = user_data[remain]
    #user_track.iloc[:,:4].apply(lambda info:locinfo2semantic(info),axis=1)

    return pd.concat([start_s.apply(locinfo2semantic,axis=1),end_s.apply(locinfo2semantic,axis=1),remain],axis=1)

def make_batches(targets,num_cores):
    n_groups = len(targets)//num_cores
    ends = [i*num_cores for i in range(1,n_groups)]
    ends.append(len(targets))
    start = 0
    batches = []

    for end in ends:
        if end>start:
            #print(start, end)
            batches.append(targets[start:end])
            start = end
    return batches

def simple_func(dt,name):
    dt = dt.reset_index()
    dt['advertiser_id'] = name
    return dt

def find_stay1(data):
    a = [[min(i),max(i)] for  i in select_continous(data.index)]
    indexes = [i[1] for i in a]
    times = [test.loc[i[1],'location_at']-test.loc[i[0],'location_at'] for i in a]
    return [indexes[i] for i in range(len(indexes)) if times[i] >= 1000]

def time_dist(data):
    return np.histogram(data['hour'],bins=range(24))[0]   

def find_stay2(data):
    a = [[min(i),max(i)] for  i in select_continous(data.index)]
    indexes = [i[1] for i in a]
    times = [test.loc[i[1],'location_at']-test.loc[i[0],'location_at'] for i in a]
    return times
