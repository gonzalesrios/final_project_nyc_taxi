#FastAPI application for NYC Taxi Fare Prediction
#It will allow users to input trip details and get fare predictions from the trained model.
#LOADS TRAINED MODEL AND SERVES PREDICTIONS VIA REST API CALLS


import numpy as np
import joblib #loads .pkl file
from fastapi import FastAPI
from pydantic import BaseModel

#----------------------
#Initialize FastAPI app
fare_app = FastAPI(title="NYC Taxi Fare Prediction API", description="API for predicting NYC taxi fares based on trip details.", version="1.0.0")


#---------------------------
#load pkl file/trained model
model = joblib.load("taxi_fare_model.pkl")


#--------------------------------------
#Define the input data model for the API
#8 features the model expects
class TripDetails(BaseModel):
    trip_distance: float
    passenger_count: int
    PULocationID: int
    DOLocationID: int
    pickup_hour: int
    pickup_dayofweek: int
    pickup_weekend: int
    pickup_month: int
    
#ENDPOINT: HEALTH CHECK
#GET /health to make sure the API is running
@fare_app.get("/health", tags=["Health Check"])
def health_check():
    """
    Health check endpoint to verify that the API is running.
    Returns a simple message indicating the API status.
    """
    return {"status": "API is running"}

#ENDPOINT: PREDICT FARE
#POST /predict_fare to get fare prediction based on trip details
@fare_app.post("/predict", tags=["Fare Prediction"])
def predict_fare(trip_details: TripDetails):
    """
    Predicts the fare for a taxi trip based on the provided trip details.
    
    Parameters:
    - trip_details: An instance of TripDetails containing the trip features.
    
    Returns:
    - A dictionary containing the predicted fare.
    """
    # Convert input data to a numpy array for prediction
    input_data = np.array([[trip_details.trip_distance,
                            trip_details.passenger_count,
                            trip_details.PULocationID,
                            trip_details.DOLocationID,
                            trip_details.pickup_hour,
                            trip_details.pickup_dayofweek,
                            trip_details.pickup_weekend,
                            trip_details.pickup_month]])
    
    # Make prediction using the loaded model
    predicted_fare = model.predict(input_data)[0]
    
    # Return the predicted fare as a JSON response
    return {
        "predicted_fare": round(float(predicted_fare), 2),
        "currency": "USD",
        "trip_distance_miles": trip_details.trip_distance,
        "status": "success",
        "model": "NYC Taxi Fare Prediction Model v1.0"
    }

#RUN LOCALLY: uvicorn main:fare_app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fare_app, host="0.0.0.0", port=8000, reload=True)
    
    
"""
I see a screen that shows my API working
Loads 3 million something rows of data
The model was trained and serves predictions through REST API calls. 
The API has a health check endpoint and a fare prediction endpoint. 
The health check endpoint returns a simple message indicating that the API is running, while the fare prediction endpoint takes in trip details and returns the predicted fare based on the trained model. 
The model expects 8 features: trip distance, passenger count, pickup location ID, dropoff location ID, pickup hour, pickup day of the week, pickup weekend indicator, and pickup month. 
The predicted fare is returned in USD along with a success status and model version information.

"""
