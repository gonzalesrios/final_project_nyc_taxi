### Final Project for COMP4450


What does this project do? 
This project is a fare estimator that predicts the fare/cost of a trip in New York City (NYC). Let's say you don't want to rent a car and are choosing between public transportation and a taxi, this app would tell you how much it would a taxi ride would cost you before you choose to take it.

What data is used and where does it come from? 
A parquet file from Janaury 2026 is used from the NYC Yellow Taxi company. The dataset contains 3.7 million real taxi trips taken during the month of January 2026.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page


What problem does it solve in real life? 
Before Uber/Lyft existed to show the estimated price, no one had any idea what a taxi would cost or if it was worth taking. This project solves that by giving an instant fare estimate.  

Who would use this?
Tourists, local New Yorkers, business travelers, and anyone deciding between taxi and public transportation. Anyone trying to maximize their budget while enjoying all NYC has to offer. 

### System Architecture
This project has multi-container application with two services working together to make it all possible. 

1. FastAPI Backend (api/)
- Receives trip details from the user
- Runs them through the trained ML model
- Returns a predicted fare amount in dollars
- Logs every prediction with a timestamp
- Runs on port 8000

2. Streamlit Frontend (monitoring/)
- User-friendly web interface
- Anyone can enter their trip details
- Displays the predicted fare clearly
- Calls the FastAPI backend automatically
- Runs on port 8501

3. Docker
- Both services run in separate Docker containers
- Containers communicate through a shared Docker network
- Can run on any computer with Docker installed

4. Machine Learning Model
- Algorithm: HistGradientBoostingRegressor
- Trained on: 3.5 million NYC taxi rides
- Features used: trip distance, passenger count, 
  pickup/dropoff location zones, time of day, day of week
- Performance: MAE = $4.06, R2 = 0.81
- Model file: api/taxi_fare_model.pkl

### Project Structure
final_project/
├── api/ - FastAPI backend service
│ ├── main.py - API endpoints
│ ├── Dockerfile -  Container for API
│ ├── requirements.txt -  Libraries for API
│ └── taxi_fare_model.pkl -  Trained model 
├── monitoring/ - Streamlit frontend
│ ├── app.py -  User interface
│ ├── Dockerfile - Container for frontend
│ └── requirements.txt - Libraries for frontend
├── tests/
│ └── test_api.py - Automated API tests
├── .github/
│ └── workflows/
│ └── ci.yml - CI/CD pipeline
├── train_model.py - Train and save the ML model
├── explore_data.py - Data exploration script
├── Makefile - Shortcuts for Docker commands
├── .gitignore - Files excluded from GitHub
└── README.md - This file

NOTE: taxi_fare_model.pkl and the parquet data file are NOT 
included in this repository — they are too large for GitHub.

### Prerequisites
Before running this project you will to make sure you have:
- Docker Desktop installed and running
- Python 3.11 installed
- Virtualization enabled in BIOS settings for Windows computer users
- The NYC taxi dataset (Janaury 2026 Parquet file). Which can be downloaded from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### How to Run it Locally
1. Create the model file
The model file is not included in this respository because it is too large for Github. However, it can be generated in your machine. It would preferable to do that first. 
Open a terminal and navigate to the project folder: cd final_project
Run: python train_model.py

That will create the 'api/taxi_fare_model.pkl'. Training the model takes about 5-15 minutes. Do not close the terminal while this is running, please. 


2. Build Docker Images
Use 'make build'
Or if you are using a Windows machine use:
docker build -t api_image ./api
docker build -t monitor_image ./monitoring

3. Run both Containers
Use 'make run'
Or if you are using a Windows machine use:
docker network create my_network
docker run -d --name api_container --network my_network -p 8000:8000 api_image
docker run -d --name monitor_container --network my_network -p 8501:8501 monitor_image

4. Open in a browser
- The interface the user sees, FrontEnd can be found at: 
http://localhost:8501
- API code and information is at: http://localhost:8000/docs

### Stop and Clean Everything
Use 'make clean'
If you are using a Windows machine use,run the following one at a time: 
docker stop api_container monitor_container
docker rm api_container monitor_container
docker rmi api_image monitor_image
docker network rm my_network
docker volume rm my_volume

### API Endpoints
Visually all the endpoints can be tested at: 
http://localhost:8000/docs

### Health Check
On your screen you should automatically see: 
{"status": "API is running"}
There's nothing to click. This message is all you need to know the API is working correclty. 

### Predict Fare 
Once you are at http://localhost:8000/predict
Fill in your trip details and click the button. 
You could also copy and past this in the POST terminal:
{-H "Content-Type: application/json"
-d '{
"trip_distance": 2.5,
"passenger_count": 1,
"PULocationID": 132,
"DOLocationID": 161,
"pickup_hour": 14,
"pickup_dayofweek": 3,
"pickup_weekend": 0,
"pickup_month": 1
}'}

What the numbers mean:
- trip_distance: 2.5 miles
- passenger_count: 1 person
- PULocationID: 132 = JFK Airport (pickup location)
- DOLocationID: 161 = Midtown Manhattan (dropoff location)
- pickup_hour: 14 = 2:00pm
- pickup_dayofweek: 3 = Thursday
- pickup_weekend: 0 = not a weekend
- pickup_month: 1 = January

You will get this response back:
```json
{
  "predicted_fare": 21.47,
  "currency": "USD",
  "status": "success",
  "model": "NYC Taxi Fare Prediction Model v1.0"
}
```
This means the estimated fare for that trip is $21.47.

### How to Run Tests
Make sure you are in the final_project folder, then run:
pytest tests/test_api/py -v

By running this command you will automatically test:
- That the API is running and healthy
- That a valid trip returns a fare prediction
- That invalid input is rejected correctly
- The predicted fare is within a realistic range of $3-$200

### CI/CD Pipeline
CI/CD stands for Continous Integration and Continous Delivery. 
In simpler words, every time you push new code to Github, it automatically runs your tests and checks your code quality. Think of it like a quality control checkpoint that runs itself without a human.

The project uses Github Actions for CI/CD. Every time the code is pushed to the main branch:
1. A fresh computer is set up in the cloud
2. Python 3.11 is instaled
3. All libraries are installed
4. flake8 chekcs the code for errors
5. pytest runs all 4 automated tests
6. Github shows green checkmarks if everything passes

### Model Performance
Take a look at how the model performed:
| Metric  | Training  | Test  | Difference  |
|-------------------------------------------|
| MAE($)  | $1.60     | $4.06 | $2.46       |
| RMSE($) | $2.88     | $7.25 | $4.35       |
| R2 SCORE| 0.9703    | 0.8098| 0.1605      |

The MAE value of $4.06, means that the average predicted fare is off by $4.06 from the actual fare. 
MAE = Mean Absolute Error = how many dollars off we are on average

Example:
- Real fare:           $20.96  (average NYC taxi fare)
- Our prediction:      $16.90  (worst case, too low)
- Our prediction:      $25.02  (worst case, too high)
- Difference:          $4.06   (this is the MAE)
So if the real taxi fare is $20.96, the app might say its anywhere between $16.90-$25.02.

The model is off by about 19% which in real life could be traffic, route taken, or accidents unaccounted for. 

R2 =     = how well the model fits the test data = how well are the fare prices
Our model has an R2 of 0.8098 which is 81/100 times the model knows why 1 fare costs more than another. 

### Notes
- taxi_fare_model.pkl not included, its too large for GitHub
- Parquet data file not included, its too large for GitHub
  Download January 2026 Yellow Taxi data from NYC TLC website
- A .gitattributes file was not included but is recommended
  in real production projects
- AI assistance (Claude by Anthropic) was used during
  development for learning and confirming the use of new methods, all used appropiately and responsibly.
- Algorithm changed from RandomForestRegressor to 
  HistGradientBoostingRegressor to reduce model file size
  from 21GB to 0.35MB so it may be uploaded to GitHub.