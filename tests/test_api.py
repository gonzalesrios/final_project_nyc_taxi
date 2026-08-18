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
    
#TEST 5: Currency is always in USD
def test_currency_is_usd():
    "Response returned should be in USD currency"
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
    assert response.json()["currency"] == "USD"
    
# TEST 6: Maximum passengers (8) is accepted
def test_maximum_passengers_accepted():
    """8 passengers is the maximum"""
    max_passengers_trip = {
        "trip_distance": 2.5,
        "passenger_count": 8,
        "PULocationID": 161,
        "DOLocationID": 234,
        "pickup_hour": 14,
        "pickup_dayofweek": 3,
        "pickup_weekend": 0,
        "pickup_month": 1
    }
    response = client.post("/predict", json=max_passengers_trip)
    assert response.status_code == 200

# TEST 8: Weekend trip is valid
def test_weekend_trip_is_valid():
    """Weekend trips (pickup_weekend=1) should be accepted"""
    weekend_trip = {
        "trip_distance": 2.5,
        "passenger_count": 1,
        "PULocationID": 161,
        "DOLocationID": 234,
        "pickup_hour": 20,
        "pickup_dayofweek": 5,
        "pickup_weekend": 1,
        "pickup_month": 1
    }
    response = client.post("/predict", json=weekend_trip)
    assert response.status_code == 200
    
# TEST 9: Long distance trip (JFK to Manhattan ~15 miles)
def test_long_distance_trip():
    """Long distance trips should return higher fare estimate"""
    long_trip = {
        "trip_distance": 15.0,
        "passenger_count": 1,
        "PULocationID": 132,
        "DOLocationID": 161,
        "pickup_hour": 14,
        "pickup_dayofweek": 3,
        "pickup_weekend": 0,
        "pickup_month": 1
    }
    response = client.post("/predict", json=long_trip)
    assert response.status_code == 200
    assert response.json()["predicted_fare"] > 0