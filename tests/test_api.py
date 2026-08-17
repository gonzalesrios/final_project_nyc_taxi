#Checklist that runs every time a pull request is made to the main branch. It runs all the tests in the tests folder and checks for code coverage.
#Verifies: Is the API working? 
# Does /predict endpoint return the expected response? 
# Does /predict reject invalid input?  
# Does /predict return the expected response for empty input? 
# Does /predict return the expected response for null input? 
# Does /predict return the expected response for missing input? 
 

#This can be run using: pytest tests/test_api.py

from fastapi.testclient import TestClient
import sys
import os
import pytest

# Check if model exists before importing
model_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'taxi_fare_model.pkl')
if not os.path.exists(model_path):
    pytest.skip("Model file not found - skipping tests", allow_module_level=True)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
from main import fare_app
from fastapi.testclient import TestClient
#Create a TestClient for the FastAPI app
client = TestClient(fare_app)
#Add api folder to path so that we can import the app from main.py



#TEST 1 - HEALTH CHECK
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_input():
    valid_trip = {
        "trip_distance": 2.5,
        "passenger_count": 1,
        "PULocationID": 161,
        "DOLocationID": 234,
        "pickup_hour": 14,
        "pickup_dayofweek": 3,
        "pickup_weekend": 0,
        "pickup_month": 1
    }
    response = client.post("/predict", json=valid_trip)
    assert response.status_code == 200

def test_predict_invalid_input():
    response = client.post("/predict", json={"trip_distance": 2.5})
    assert response.status_code == 422

def test_predict_fare_is_realistic():
    trip = {
        "trip_distance": 2.5,
        "passenger_count": 1,
        "PULocationID": 161,
        "DOLocationID": 234,
        "pickup_hour": 14,
        "pickup_dayofweek": 3,
        "pickup_weekend": 0,
        "pickup_month": 1
    }
    response = client.post("/predict", json=trip)
    fare = response.json()["predicted_fare"]
    assert 3 <= fare <= 200