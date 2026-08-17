# This is the Dockerfile for the API service

#Load the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

#Copy the requirements file into the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the rest of the application code into the container
COPY main.py .
COPY taxi_fare_model.pkl .

# Expose the port that the API will run on
EXPOSE 8000

#Run the API service using uvicorn
CMD ["uvicorn", "main:fare_app", "--host", "0.0.0.0", "--port", "8000"]
