#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, adjusted_rand_score, accuracy_score
from sklearn.mixture import GaussianMixture
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

def plot_clusters(model, n_clusters, x, y):
    kmeans=KMeans(n_clusters=n_clusters,init='k-means++',random_state=0)
    y_kmeans=kmeans.fit_predict(data)

    colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown", "Gray", "Cyan", "Magenta", "Lime", "Teal", "Lavender", "Maroon", "Olive", "Navy", "Aquamarine"]
    
    #plotting the clusters
    fig,ax = plt.subplots(figsize=(14,6))
    for i in range(0, n_clusters):
        ax.scatter(data[y_kmeans==i,0],data[y_kmeans==i,1],s=100,c=colors[i],label=f'Cluster {i+1}')
    
    ax.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='black',label='Centroid')
    plt.title(f'Cluster Segmentation of Customers for {i+1} clusters')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()


# In[2]:


customers_df = pd.read_csv('data/Mall_Customers.csv')
customers_df.sample(10)


# In[3]:


fig = px.scatter_3d(customers_df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)', color='Gender')
fig.update_layout(scene = dict(
                    xaxis_title='Age',
                    yaxis_title='Annual Income',
                    zaxis_title='Spending score'),
                  width=800, height=600)
fig.show()


# In[4]:


fig = px.scatter(customers_df, x='Annual Income (k$)', y='Spending Score (1-100)', color='Age')
fig.update_layout(scene = dict(
                    xaxis_title='Annual Income',
                    yaxis_title='Spending score'))
fig.show()

fig = px.scatter(customers_df, x='Spending Score (1-100)', y='Annual Income (k$)', color='Age')
fig.update_layout(scene = dict(
                    yaxis_title='Annual Income',
                    xaxis_title='Spending score'))
fig.show()

fig = px.scatter(customers_df, x='Age', y='Annual Income (k$)', color='Spending Score (1-100)')
fig.update_layout(scene = dict(
                    yaxis_title='Annual Income',
                    xaxis_title='Age'))
fig.show()

fig = px.scatter(customers_df, x='Age', y='Spending Score (1-100)', color='Annual Income (k$)')
fig.update_layout(scene = dict(
                    yaxis_title='Spending Score',
                    xaxis_title='Age'))
fig.show()


# In[5]:


# k-means clustering based on annual income
data = customers_df.iloc[:,[3,4]].values

E=[] # Euclidian MSE for each point
for i in range(1,11):
    income_kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    income_kmeans.fit_predict(data)
    E.append(income_kmeans.inertia_) #inertia_ = to find the MSE value

plt.plot(range(1,11),E)
plt.title('MSE by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('MSE')
plt.show()


# In[6]:


plot_clusters(income_kmeans,5, 'Annual Income', 'Spending Score')


# In[7]:


# k-means clustering based on Age¶
data = customers_df.iloc[:,[2,4]].values

E=[] # Euclidian MSE for each point
for i in range(1,11):
    age_kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
    age_kmeans.fit_predict(data)
    E.append(age_kmeans.inertia_) #inertia_ = to find the MSE value

plt.plot(range(1,11),E)
plt.title('MSE by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('MSE')
plt.show()


# In[8]:


plot_clusters(age_kmeans,4, 'Age', 'Spending Score')


# In[9]:


# GMM clustering based on Annual Income¶
data = customers_df.iloc[:,[3,4]].values

BIC = [] # Stockage des valeurs de BIC
AIC = [] # Stockage des valeurs de l'AIC

for i in range(1,11):
    income_gmm = GaussianMixture(n_components=i)
    income_gmm.fit(data)
    BIC.append(income_gmm.bic(data))
    AIC.append(income_gmm.aic(data))

# Tracé du graphique BIC
plt.plot(range(1,11), BIC, marker='o')
plt.title('BIC by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC')
plt.show()

# Tracé du graphique AIC
plt.plot(range(1, 11), AIC, marker='o')
plt.title('AIC by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('AIC')
plt.show()


# In[10]:


plot_clusters(income_gmm,5, 'Annual Income', 'Spending Score')


# In[11]:


# GMM clustering based on Age¶
data = customers_df.iloc[:,[2,4]].values

BIC = [] # Stockage des valeurs de BIC
AIC = [] # Stockage des valeurs de l'AIC

for i in range(1,11):
    age_gmm = GaussianMixture(n_components=i)
    age_gmm.fit(data)
    BIC.append(age_gmm.bic(data))
    AIC.append(age_gmm.aic(data))

# Tracé du graphique BIC
plt.plot(range(1,11), BIC, marker='o')
plt.title('BIC by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC')
plt.show()

# Tracé du graphique AIC
plt.plot(range(1, 11), AIC, marker='o')
plt.title('AIC by Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('AIC')
plt.show()


# In[12]:


plot_clusters(age_gmm,3, 'Age', 'Spending Score')


# In[13]:


with open('../backend/endpoints/models/income_kmeans_model.pkl', 'wb') as f:
    pickle.dump(income_kmeans, f)

with open('../backend/endpoints/models/age_kmeans_model.pkl', 'wb') as f:
    pickle.dump(age_kmeans, f)

with open('../backend/endpoints/models/income_gmm_model.pkl', 'wb') as f:
    pickle.dump(income_gmm, f)

with open('../backend/endpoints/models/age_gmm_model.pkl', 'wb') as f:
    pickle.dump(age_gmm, f)

