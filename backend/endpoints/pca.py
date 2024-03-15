import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
import pickle


def pca_predict():
    """
    Load the Mall Customers dataset, apply PCA and KMeans clustering, 
    and return the original data and a base64 plot.
    """
    data = pd.read_csv("data/Mall_Customers.csv")
    data_numeric = data.drop(["CustomerID", "Gender"], axis=1)

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    # Load PCA model
    with open("models/pca.pkl", "rb") as file:
        pca = pickle.load(file)

    # Apply PCA transformation
    principal_components = pca.transform(data_scaled)
    pca_df = pd.DataFrame(principal_components, columns=["PC1", "PC2"])

    # Load KMeans model
    with open("../models/kmeans.pkl", "rb") as file:
        kmeans = pickle.load(file)

    # Predict clusters
    cluster_labels = kmeans.predict(data_scaled)
    pca_df["Cluster"] = cluster_labels

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df["PC1"], pca_df["PC2"],
                c=pca_df["Cluster"], cmap="viridis", edgecolor="k", s=100)
    plt.title("PCA Clustering on Mall Customers Data")
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

    return {"data": data.to_dict(), "plot_base64": plot_base64}
