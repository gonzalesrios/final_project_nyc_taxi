#APIT tests to verify API

#https://code.visualstudio.com/api/extension-guides/testing
#https://code.visualstudio.com/api/extension-guides/testing
#https://www.geeksforgeeks.org/software-testing/test-cases-for-api-testing/


from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

#Add API folder to path
sys.path.append(os.path.dirname(__file__))

#Mock model Before importing to main
#The real model is in .pkl format and not found in Github
with patch('joblib.load') as mock_load:
    mock_model = MagicMock()
    mock_model.predict.return_value = [21.47]
    mock_load.return_value = mock_model
    from main import fare_app

#Create test client
client = TestClient(fare_app)

# TEST 1: Is the API alive and healthy?
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

# TEST 2: Does /predict work?
def test_predict_valid_trip():
    "Check /predict returns a fare when valid input is inputted"
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
    assert response.status_code == 200
    assert "predicted_fare" in response.json()

# TEST 3: Does /predict reject empty request?
def test_predict_rejects_empty_request():
    response = client.post("/predict", json={})
    assert response.status_code == 422