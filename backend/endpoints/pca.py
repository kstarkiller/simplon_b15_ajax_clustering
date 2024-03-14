import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
import pickle


def pca_predict():
    # Load the training data
    data = pd.read_csv("../data/Mall_Customers.csv")

    # Extract features for clustering
    data_numeric = data.drop(["CustomerID", "Gender"], axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])

    # Perform clustering on training data
    with open("../models/pca.pkl", "rb") as file:
        kmeans = pickle.load(file)

    cluster_labels = kmeans.predict(data_scaled)
    pca_df["Cluster"] = cluster_labels

    # Generate plot on training data
    plt.figure(figsize=(10, 6))
    plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"],
        cmap="viridis",
        edgecolor="k",
        s=100,
    )
    plt.title("PCA Clustering on Training Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Cluster")
    plt.grid(True)

    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Return the training data and the base64 encoded plot
    return {"data": data.to_dict(), "plot_base64": plot_base64}
