from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints.pca import pca_predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/data_and_plot/")
async def data_and_plot():
    return pca_predict()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
