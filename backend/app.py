from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from endpoints.pca import pca_predict
from endpoints.kmeans_and_gmm import (
    income_kmeans_predict,
    age_kmeans_predict,
    income_gmm_predict,
    age_gmm_predict,
)  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/plot_pca/")
async def plot_pca():
    return pca_predict()


@app.get("/plot_income_kmeans/")
async def plot_income_kmeans():
    return income_kmeans_predict()


@app.get("/plot_age_kmeans/")
async def plot_age_kmeans():
    return age_kmeans_predict()


@app.get("/plot_income_gmm/")
async def plot_income_gmm():
    return income_gmm_predict()


@app.get("/plot_age_gmm/")
async def plot_age_gmm():
    return age_gmm_predict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
