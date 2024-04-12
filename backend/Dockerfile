# FROM python:3.11-alpine
FROM python:3.10-slim

# Install Python packages
RUN pip install --no-cache-dir pandas matplotlib scikit-learn fastapi uvicorn

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]