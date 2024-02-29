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


# In[2]:


def user_process(user_data):
    user_data = user_data.sort_values(by='location_at')
    user_data = user_data.set_index('time')
    #user_data = user_data.resample('5T').first()#important
    user_data = user_data.reset_index()
    user_data = user_data.dropna(subset=['latitude'])
    user_data = user_data.reset_index().drop(columns = ['index'])
    return user_data
    


# In[3]:


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


# In[4]:


def plot_user(user_data,fill=False):
    if fill:
        user_data.label = user_data.label.fillna(0)
    user_data[['latitude','longitude']].plot.scatter(y='latitude',x='longitude',c=user_data.label,cmap= "plasma",s=0.1)


# In[5]:


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
#[labels != -1]
#user_data[user_data['speed']<0.5][['latitude','longitude']].plot.scatter(y='latitude',x='longitude',c=labels,cmap= "plasma",s=0.1)
#user_data[user_data['speed']<0.5][labels != -1][['latitude','longitude']]


# In[6]:


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


# In[7]:


#action
def cal_dd(user_data):
    tempdf = pd.concat([user_data[['latitude','longitude']].shift(),user_data[['latitude','longitude']]],axis=1)
    r = tempdf.apply(lambda row: haversine_distance(row[1], row[0], row[3], row[2]),axis=1)
    return r


# In[8]:


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
                


# In[9]:


def find_stay1(data):# find the last index of a stay in a continuous time
    a = [[min(i),max(i)] for  i in select_continous(data.index)]
    indexes = [i[1] for i in a]#last one of the period
    times = [data.loc[i[1],'location_at']-data.loc[i[0],'location_at'] for i in a]# corresponding time period of a stay
    return [indexes[i] for i in range(len(indexes)) if times[i] >= (3600*2)]


# In[10]:


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


# In[11]:


#def find_end(user_data):
#natural
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


# In[13]:


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


# In[14]:


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
        


# In[15]:


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


# In[16]:


rename = pd.read_csv('rename.csv',index_col=0).squeeze()
rename = rename.fillna('')
rdict = rename.to_dict()


# In[17]:


with open('boston_final.json') as f:
    venue_data = json.load(f)
venue_data = pd.DataFrame(venue_data)
#decrease types and delete nonsense place
venue_data.types = venue_data.types.apply(lambda x:[rdict[i] for i in x])
venue_data.types = venue_data.types.apply(lambda x:[i for i in x if i!=''])
venue_data.types = venue_data.types.apply(lambda x:list(set(x)))
venue_data = venue_data[venue_data.types.apply(len)>0]
semantics = list(np.unique([i for s in venue_data.types for i in s]))+['home','poi']


# In[18]:


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


# In[19]:


targets = os.listdir('./user_data1')
done = os.listdir('./user_data')
problematic = ['1000_1100.csv','3600_3700.csv','4100_4200.csv']
targets = [i for i in targets if i not in done]
#targets = [i for i in targets if i not in problematic]


# In[20]:


targets


# In[ ]:


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


# In[ ]:


sep


# In[21]:


#targets = os.listdir('./user_data1')
in_path = './user_data1/'+targets[0]
print(in_path)


# In[22]:


sample = pd.read_csv(in_path,index_col=0)
sample = sample_process(sample)


# In[25]:


users = np.unique(sample['advertiser_id'])
problems = ['94030A83-26C2-4F02-B6D6-898C769D6AA8','84B42011-9717-4855-803F-5043A984D447','245D3C52-1917-4743-982A-06BCA2E27203','2825C708-7307-4D1E-ACCC-082D8EA273FA']
users = [i for i in users if i not in problems]
dfs = [user_transform(sample[sample['advertiser_id']==user],verbose=True) for user in users]


# In[26]:


def simple_func(dt,name):
    dt = dt.reset_index()
    dt['advertiser_id'] = name
    return dt


# In[27]:


resample = pd.concat([simple_func(dfs[i],users[i]) for i in range(len(users))])


# In[28]:


out_path = './user_data/'+targets[0]
resample.to_csv(out_path)


# In[66]:


import time 
s = time.time()
df = sample.groupby('advertiser_id').apply(user_transform)
e = time.time()
print(e-s)
#df = sample.groupby('advertiser_id').apply(user_transform)


# In[ ]:


sep


# In[63]:


np.max(user_tracks(user_data_s).end_speed)


# In[136]:


fig = plt.figure()
ax1 = fig.add_subplot(111)
#user_data_s[['latitude','longitude']].plot.scatter(y='latitude',x='longitude',s=0.5,ax=ax1)
user_data_s[user_data_s['action']==1][['latitude','longitude']].plot.scatter(y='latitude',x='longitude',s=1,c='red',ax=ax1)
ax1.scatter(y=home[0],x=home[1],c='orange',s=0.5)


# In[42]:


user_data.loc[:100].to_csv('eg.csv')


# In[17]:


action = 16
test = user_data[user_data['action']==action]
dbscan = DBSCAN(eps = 1*(10**-3), min_samples = 3).fit(test[['latitude','longitude']])
labels = dbscan.labels_
test[['latitude','longitude']].plot.scatter(y='latitude',x='longitude',c=labels,cmap= "plasma",s=1)
#test[['latitude','longitude']][:].plot.scatter(y='latitude',x='longitude',c='blue',s=0.5)
test['labels']=labels
#test.groupby('labels').apply(find_stay1)

#
def find_stay1(data):
    a = [[min(i),max(i)] for  i in select_continous(data.index)]
    indexes = [i[1] for i in a]
    times = [test.loc[i[1],'location_at']-test.loc[i[0],'location_at'] for i in a]
    return [indexes[i] for i in range(len(indexes)) if times[i] >= 1000]
grouped = test.groupby('labels').apply(find_stay1)
grouped = grouped[grouped.index>=0]
grouped = [i for s in grouped for i in s]
grouped
#test.iloc([a[0][0]]
#find_stay1(r1)


# In[116]:


def time_dist(data):
    return np.histogram(data['hour'],bins=range(24))[0]    


# In[117]:


#time analysis
#all
time_dist(user_data)
#diff patterns of week and weekend
user_data.groupby('weekend').apply(time_dist)
#stable in month
#stable in week
#outlier detect and remove


# In[48]:


atts = [ 'latitude', 'longitude', 'horizontal_accuracy',  'speed',
       'hour','weekend']
first = user_data_s.groupby('action')[atts].first()
last = 


# In[12]:


def find_stay2(data):
    a = [[min(i),max(i)] for  i in select_continous(data.index)]
    indexes = [i[1] for i in a]
    times = [test.loc[i[1],'location_at']-test.loc[i[0],'location_at'] for i in a]
    return times
r1 = test[test['labels']==0]
#r1
find_stay2(r1)[0]/3600
#r1


# In[118]:


user_data.groupby('month').apply(time_dist)


# In[119]:


user_data.groupby('week').apply(time_dist)


# In[134]:


user_data['home_dis'] = user_data[['latitude','longitude']].apply(lambda row: haversine_distance(home[1], home[0], row[1], row[0]),axis=1)
user_data['home'] = (user_data['home_dis']<0.2).astype(int)
user_data['POI_dis'] = user_data[['latitude','longitude']].apply(lambda row: min([haversine_distance(poi[1], poi[0], row[1], row[0]) for poi in POI]),axis=1)
user_data['POI'] = (user_data['POI_dis']<0.2).astype(int)
color = user_data['home'] + user_data['POI']*0.5
user_data[['latitude','longitude']].plot.scatter(y='latitude',x='longitude',c=color,cmap= "plasma",s=0.05)


# In[ ]:




