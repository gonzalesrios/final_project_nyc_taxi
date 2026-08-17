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

#Add api folder to path so that we can import the app from main.py
sys.path.append(os.path.join(os.path.dirname(__file__), '../api'))

from main import fare_app

#Create a TestClient for the FastAPI app
client = TestClient(fare_app)

#TEST 1 - HEALTH CHECK
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
