import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import base64
from io import BytesIO
from sklearn.cluster import KMeans


def income_kmeans_predict(n_clusters=5):
    script_dir = os.path.dirname(__file__)  # get the directory of the current script
    file_path = os.path.join(
        script_dir, "data/Mall_Customers.csv"
    )  # join the script directory with the relative file path
    df = pd.read_csv(file_path)
    df.sample(10)

    # k-means clustering based on annual income
    data = df.iloc[:, [3, 4]].values

    income_kmeans = KMeans(n_clusters, init="k-means++", random_state=0, n_init=10)
    y = income_kmeans.fit_predict(data)
    mse = income_kmeans.inertia_  # inertia_ = to find the MSE value

    df["Cluster"] = y
    # plotting the clusters
    plt.subplots(figsize=(10, 6))
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"],
        cmap="viridis",
        edgecolor="k",
        s=100,
    )

    plt.scatter(
        income_kmeans.cluster_centers_[:, 0],
        income_kmeans.cluster_centers_[:, 1],
        s=200,
        c="black",
        label="Centroid",
    )
    plt.title(f"Cluster Segmentation of Customers for {n_clusters} clusters")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.grid(True)

    # Set the background color to transparent
    fig = plt.gcf()
    fig.patch.set_alpha(0)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return {
        "data": df.to_dict(),
        "clusters": n_clusters,
        "MSE": mse,
        "plot_base64": plot_base64,
    }


def age_kmeans_predict(n_clusters=5):
    script_dir = os.path.dirname(__file__)  # get the directory of the current script
    file_path = os.path.join(
        script_dir, "data/Mall_Customers.csv"
    )  # join the script directory with the relative file path
    df = pd.read_csv(file_path)
    df.sample(10)

    # k-means clustering based on annual income
    data = df.iloc[:, [2, 4]].values

    age_kmeans = KMeans(n_clusters, init="k-means++", random_state=0)
    y = age_kmeans.fit_predict(data)
    mse = age_kmeans.inertia_  # inertia_ = to find the MSE value

    colors = [
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Purple",
        "Orange",
        "Pink",
        "Brown",
        "Gray",
        "Cyan",
        "Magenta",
        "Lime",
        "Teal",
        "Lavender",
        "Maroon",
        "Olive",
        "Navy",
        "Aquamarine",
    ]

    # plotting the clusters
    plt.subplots(figsize=(10, 6))
    for i in range(0, n_clusters):
        plt.scatter(
            data[y == i, 0], data[y == i, 1], s=100, c=colors[i], label=f"Cluster {i+1}"
        )

    plt.scatter(
        age_kmeans.cluster_centers_[:, 0],
        age_kmeans.cluster_centers_[:, 1],
        s=200,
        c="black",
        label="Centroid",
    )
    plt.title(f"Cluster Segmentation of Customers for {i+1} clusters")
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.grid(True)

    # Set the background color to transparent
    fig = plt.gcf()
    fig.patch.set_alpha(0)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return {
        "data": df.to_dict(),
        "clusters": n_clusters,
        "MSE": mse,
        "plot_base64": plot_base64,
    }


def income_gmm_predict(n_clusters=5):
    script_dir = os.path.dirname(__file__)  # get the directory of the current script
    file_path = os.path.join(
        script_dir, "data/Mall_Customers.csv"
    )  # join the script directory with the relative file path
    df = pd.read_csv(file_path)
    df.sample(10)

    # GMM clustering based on Annual Income¶
    data = df.iloc[:, [3, 4]].values

    income_gmm = GaussianMixture(n_components=n_clusters)
    y = income_gmm.fit_predict(data)
    bic = income_gmm.bic(data)  # Stockage du BIC
    aic = income_gmm.aic(data)  # Stockage de l'AIC

    colors = [
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Purple",
        "Orange",
        "Pink",
        "Brown",
        "Gray",
        "Cyan",
        "Magenta",
        "Lime",
        "Teal",
        "Lavender",
        "Maroon",
        "Olive",
        "Navy",
        "Aquamarine",
    ]

    # plotting the clusters
    plt.subplots(figsize=(10, 6))
    for i in range(0, n_clusters):
        plt.scatter(
            data[y == i, 0], data[y == i, 1], s=100, c=colors[i], label=f"Cluster {i+1}"
        )

    plt.title(f"Cluster Segmentation of Customers for {i+1} clusters")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.grid(True)

    # Set the background color to transparent
    fig = plt.gcf()
    fig.patch.set_alpha(0)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return {
        "data": df.to_dict(),
        "clusters": n_clusters,
        "BIC": bic,
        "AIC": aic,
        "plot_base64": plot_base64,
    }


def age_gmm_predict(n_clusters=5):
    script_dir = os.path.dirname(__file__)  # get the directory of the current script
    file_path = os.path.join(
        script_dir, "data/Mall_Customers.csv"
    )  # join the script directory with the relative file path
    df = pd.read_csv(file_path)
    df.sample(10)

    # GMM clustering based on Annual Income¶
    data = df.iloc[:, [2, 4]].values

    age_gmm = GaussianMixture(n_components=n_clusters)
    y = age_gmm.fit_predict(data)
    bic = age_gmm.bic(data)  # Stockage du BIC
    aic = age_gmm.aic(data)  # Stockage de l'AIC

    df["Cluster"] = y
    # plotting the clusters
    plt.subplots(figsize=(10, 6))
    plt.scatter(
        df["Age"],
        df["Spending Score (1-100)"],
        c=df["Cluster"],
        cmap="viridis",
        edgecolor="k",
        s=100,
    )

    plt.title(f"Cluster Segmentation of Customers for {n_clusters} clusters")
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.grid(True)

    # Set the background color to transparent
    fig = plt.gcf()
    fig.patch.set_alpha(0)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return {
        "data": df.to_dict(),
        "clusters": n_clusters,
        "BIC": bic,
        "AIC": aic,
        "plot_base64": plot_base64,
    }
