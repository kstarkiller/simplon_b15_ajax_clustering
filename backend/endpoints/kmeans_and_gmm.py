import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import base64
from io import BytesIO

def income_kmeans_predict(n_clusters=5):
    df = pd.read_csv('data/Mall_Customers.csv')
    df.sample(10)

    # k-means clustering based on annual income
    data = df.iloc[:,[3,4]].values

    income_kmeans=KMeans(n_clusters, init='k-means++',random_state=0)
    income_kmeans.fit_predict(data)
    mse = income_kmeans.inertia_ #inertia_ = to find the MSE value

    colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown", "Gray",
              "Cyan", "Magenta", "Lime", "Teal", "Lavender", "Maroon", "Olive", "Navy", "Aquamarine"]

    #plotting the clusters
    fig, ax = plt.subplots(figsize=(14,6))
    for i in range(0, n_clusters):
        ax.scatter(data[income_kmeans==i,0],data[income_kmeans==i,1],s=100,c=colors[i],label=f'Cluster {i+1}')
    
    ax.scatter(income_kmeans.cluster_centers_[:,0],income_kmeans.cluster_centers_[:,1],s=200,c='black',label='Centroid')
    plt.title(f'Cluster Segmentation of Customers for {i+1} clusters')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return {'data': df.to_dict(),
            'clusters': n_clusters,
            'MSE': mse,
            'plot_base64': plot_base64}


def age_kmeans_predict(n_clusters=5):
    df = pd.read_csv('data/Mall_Customers.csv')
    df.sample(10)

    # k-means clustering based on annual income
    data = df.iloc[:,[2,4]].values

    age_kmeans=KMeans(n_clusters, init='k-means++',random_state=0)
    age_kmeans.fit_predict(data)
    mse = age_kmeans.inertia_ #inertia_ = to find the MSE value
    
    colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown", "Gray",
              "Cyan", "Magenta", "Lime", "Teal", "Lavender", "Maroon", "Olive", "Navy", "Aquamarine"]

    #plotting the clusters
    fig, ax = plt.subplots(figsize=(14,6))
    for i in range(0, n_clusters):
        ax.scatter(data[age_kmeans==i,0],data[age_kmeans==i,1],s=100,c=colors[i],label=f'Cluster {i+1}')
    
    ax.scatter(age_kmeans.cluster_centers_[:,0],age_kmeans.cluster_centers_[:,1],s=200,c='black',label='Centroid')
    plt.title(f'Cluster Segmentation of Customers for {i+1} clusters')
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return {'data': df.to_dict(),
            'clusters': n_clusters,
            'MSE': mse,
            'plot_base64': plot_base64}


def income_gmm_predict(n_clusters=5):
    df = pd.read_csv('data/Mall_Customers.csv')
    df.sample(10)

    # GMM clustering based on Annual Income¶
    data = df.iloc[:,[3,4]].values

    income_gmm = GaussianMixture(n_components=n_clusters)
    income_gmm.fit(data)
    bic = income_gmm.bic(data) # Stockage du BIC
    aic = income_gmm.aic(data) # Stockage de l'AIC

    colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown", "Gray",
              "Cyan", "Magenta", "Lime", "Teal", "Lavender", "Maroon", "Olive", "Navy", "Aquamarine"]

    #plotting the clusters
    fig, ax = plt.subplots(figsize=(14,6))
    for i in range(0, n_clusters):
        ax.scatter(data[income_gmm==i,0],data[income_gmm==i,1],s=100,c=colors[i],label=f'Cluster {i+1}')
    
    plt.title(f'Cluster Segmentation of Customers for {i+1} clusters')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return {'data': df.to_dict(),
            'clusters': n_clusters,
            'BIC': bic,
            'AIC': aic,
            'plot_base64': plot_base64}


def age_gmm_predict(n_clusters=5):
    df = pd.read_csv('data/Mall_Customers.csv')
    df.sample(10)

    # GMM clustering based on Annual Income¶
    data = df.iloc[:,[2,4]].values

    age_gmm = GaussianMixture(n_components=n_clusters)
    age_gmm.fit(data)
    bic = age_gmm.bic(data) # Stockage du BIC
    aic = age_gmm.aic(data) # Stockage de l'AIC

    colors = ["Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Brown", "Gray",
              "Cyan", "Magenta", "Lime", "Teal", "Lavender", "Maroon", "Olive", "Navy", "Aquamarine"]

    #plotting the clusters
    fig, ax = plt.subplots(figsize=(14,6))
    for i in range(0, n_clusters):
        ax.scatter(data[age_gmm==i,0],data[age_gmm==i,1],s=100,c=colors[i],label=f'Cluster {i+1}')
    
    plt.title(f'Cluster Segmentation of Customers for {i+1} clusters')
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return {'data': df.to_dict(),
            'clusters': n_clusters,
            'BIC': bic,
            'AIC': aic,
            'plot_base64': plot_base64}