#Checklist that runs every time a pull request is made to the main branch. It runs all the tests in the tests folder and checks for code coverage.
#Verifies: Is the API working? 
# Does /predict endpoint return the expected response? 
# Does /predict reject invalid input?  
# Does /predict return the expected response for empty input? 
# Does /predict return the expected response for null input? 
# Does /predict return the expected response for missing input? 
 

#This can be run using: pytest tests/test_api.py

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

# Mock the model BEFORE importing main
# This prevents the "model file not found" error in CI
with patch('joblib.load') as mock_load:
    mock_model = MagicMock()
    mock_model.predict.return_value = [21.47]
    mock_load.return_value = mock_model
    from main import fare_app
    
#Create a TestClient for the FastAPI app
client = TestClient(fare_app)
#Add api folder to path so that we can import the app from main.py



# TEST 1: Health check
def test_health_check():
    """API should be running and healthy"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

# TEST 2: Valid prediction returns 200
def test_predict_valid_input():
    """Valid trip details should return a prediction"""
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
    assert "predicted_fare" in response.json()

# TEST 3: Missing fields returns error
def test_predict_invalid_input():
    """Missing required fields should return 422 error"""
    response = client.post("/predict", json={"trip_distance": 2.5})
    assert response.status_code == 422

# TEST 4: Fare is positive number
def test_predict_returns_positive_fare():
    """Predicted fare should always be a positive number"""
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
    assert response.json()["predicted_fare"] > 0