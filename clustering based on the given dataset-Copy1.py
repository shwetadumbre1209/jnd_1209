#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[3]:


#import data
df = pd.read_csv(r'C:\Users\Admin\Desktop\extractedfeatures1.csv')
print(df)


# In[4]:


#Standardise the data 
from sklearn.preprocessing import StandardScaler
x = df.iloc[:, [0,1,2,3,4,5]].values
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
Clus_dataSet


# In[5]:


#k mean clustering k=3
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(x)
labels = k_means.labels_
print(labels)


# In[6]:


#To assign labels to each row
df["Clus_km"] = labels
df.head(32)


# In[7]:


# check the centroid values by averaging the features in each cluster
df.groupby('Clus_km').mean()


# In[8]:


#the distribution of people based on step length and stride length:
area = np.pi * ( x[:, 1])**2  
plt.scatter(x[:, 0], x[:, 1], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Step Length(cm)', fontsize=18)
plt.ylabel('Stride Length(cm)', fontsize=16)

plt.show()


# In[10]:


#Standardise the data and consider step length, stride length, step time, stride time features
from sklearn.preprocessing import StandardScaler
x = df.iloc[:, [0,1,2,3]].values
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
Clus_dataSet


# In[9]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('step time', fontsize=18)
# plt.xlabel('step length', fontsize=16)
# plt.zlabel('stride length', fontsize=16)
ax.set_xlabel('Step Length(cm)')
ax.set_ylabel('Step time(sec)')
ax.set_zlabel('Stride Length(cm)')

ax.scatter(x[:, 0], x[:, 1], x[:, 2], c= labels.astype(np.float))


# In[11]:


#k mean clustering k=3
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(x)
labels = k_means.labels_
print(labels)


# In[14]:


k_means.cluster_centers_


# In[15]:


#Standardise the data and consider step length, stride length, step time, stride time, cadence features
from sklearn.preprocessing import StandardScaler
x = df.iloc[:, [0,1,2,3,4]].values
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
Clus_dataSet


# In[16]:


#k mean clustering k=3
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(x)
labels = k_means.labels_
print(labels)


# In[17]:


k_means.cluster_centers_


# In[18]:


#Standardise the data and consider step length, stride length features
from sklearn.preprocessing import StandardScaler
x = df.iloc[:, [0,1]].values
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
Clus_dataSet


# In[19]:


#k mean clustering k=3
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(x)
labels = k_means.labels_
print(labels)


# In[20]:


k_means.cluster_centers_


# In[29]:


#Standardise the data 
from sklearn.preprocessing import StandardScaler
x = df.iloc[:, [0,1,2,3,4,5]].values
x = np.nan_to_num(x)
Clus_dataSet = StandardScaler().fit_transform(x)
Clus_dataSet


# In[30]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('step time', fontsize=18)
# plt.xlabel('step length', fontsize=16)
# plt.zlabel('stride length', fontsize=16)
ax.set_xlabel('Step Length(cm)')
ax.set_ylabel('Step time(sec)')
ax.set_zlabel('Stride Length(cm)')

ax.scatter(x[:, 0], x[:, 1], x[:, 2], c= labels.astype(np.float))


# In[32]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Stride Length(cm)')
ax.set_ylabel('Step length(cm)')
ax.set_zlabel('Stride time(sec)')

ax.scatter(x[:, 1], x[:, 2], x[:, 3], c= labels.astype(np.float))


# In[33]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.set_xlabel('Stride time')
ax.set_ylabel('cadence')
ax.set_zlabel('velocity')

ax.scatter(x[:, 3], x[:, 4], x[:, 5], c= labels.astype(np.float))


# In[ ]:




