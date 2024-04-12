# FROM python:3.11-alpine
FROM devashishupadhyay/scikit-learn-docker

# Install build dependencies including C and C++ compilers
# RUN apk add --no-cache --virtual .build-deps gcc g++ musl-dev python3-dev libffi-dev openssl-dev
# RUN apk add --no-cache libgomp

# Install Python packages
# RUN pip install --no-cache-dir pandas matplotlib scikit-learn fastapi uvicorn
RUN pip install --no-cache-dir fastapi uvicorn


# Clean up build dependencies to reduce image size
# RUN apk del .build-deps

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]