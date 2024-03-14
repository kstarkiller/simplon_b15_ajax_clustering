# API with PCA Clustering

## Usage

1. Start the API server:

   ```
   cd backend
   ```

   ```bash
   uvicorn main:app --reload
   ```

2. Open your web browser and navigate to `http://localhost:8000/data_and_plot/` to view the clustered data plot.

## API Endpoints

### GET /data_and_plot/

This endpoint returns the training data and a base64 encoded plot of the clustered data.
