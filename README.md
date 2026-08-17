### Final Project for COMP4450


What does this project do? 
This project is a fare estimator that predicts the fare/cost of a trip in New York City (NYC). Let's say you don't want to rent a car and are choosing between public transportation and a taxi, this app would tell you how much it would a taxi ride would cost you before you choose to take it.

What data is used and where does it come from? 
A parquet file from January 2026 is used from the NYC Yellow Taxi company. The dataset contains 3.7 million real taxi trips taken during the month of January 2026.

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
|-- api/ - FastAPI backend service
│ |-- main.py - API endpoints
│ |-- Dockerfile -  Container for API
│ |-- requirements.txt -  Libraries for API
│ |-- taxi_fare_model.pkl -  Trained model 
|-- monitoring/ - Streamlit frontend
│ |-- app.py -  User interface
│ |-- Dockerfile - Container for frontend
│ └── requirements.txt - Libraries for frontend
|-- tests/
│ └── test_api.py - Automated API tests
|-- .github/
│ └── workflows/
│ └── ci.yml - CI/CD pipeline
|-- train_model.py - Train and save the ML model
|-- explore_data.py - Data exploration script
|-- Makefile - Shortcuts for Docker commands
|-- .gitignore - Files excluded from GitHub
|-- README.md - This file

NOTE: taxi_fare_model.pkl and the parquet data file are NOT 
included in this repository — they are too large for GitHub.

### Prerequisites
Before running this project you will to make sure you have:
- Docker Desktop installed and running
- Python 3.11 installed
- Virtualization enabled in BIOS settings for Windows computer users
- The NYC taxi dataset (January 2026 Parquet file). Which can be downloaded from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### How to Run it Locally
1. Create the model file
The model file is not included in this repository because it is too large for Github. However, it can be generated in your machine. It would preferable to do that first. 
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
pytest tests/test_api.py -v

By running this command you will automatically test:
- That the API is running and healthy
- That a valid trip returns a fare prediction
- That invalid input is rejected correctly
- The predicted fare is within a realistic range of $3-$200

### CI/CD Pipeline
CI/CD stands for Continuous Integration and Continouus Delivery. 
In simpler words, every time you push new code to Github, it automatically runs your tests and checks your code quality. Think of it like a quality control checkpoint that runs itself without a human.

The project uses Github Actions for CI/CD. Every time the code is pushed to the main branch:
1. A fresh computer is set up in the cloud
2. Python 3.11 is installed
3. All libraries are installed
4. flake8 checks the code for errors
5. pytest runs all 4 automated tests
6. Github shows green checkmarks if everything passes

### Model Performance
Take a look at how the model performed:
| Metric  | Training  | Test  | Difference  |
|---------|-----------|-------|-------------|
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

### AWS Deployment Guide
EC2 (Elastic Compute Cloud) is like renting a computer from Amazon that lives in the cloud. The app runs on Amazon's computer 24/7. 
What to do:
Before starting anything select a method to access AWS

Option A: Launch an EC2 Instance with Regular AWS Account
- Log into AWS Console at https://aws.amazon.com
- Click "Create an AWS Account" or Log-In if you already have an active account. 
- You will need a Credit Card (CC) to sign up
- New accounts get 12 months of free tier access
- Read the website for all details, and promotions


Option B: Launch an EC2 Instance with AWS Academy (for students)
- Log into Canvas
- Click Modules
- Click "Launch AWS Academy Learner Lab"
- Click "Start Lab" in the top right corner
- Wait until the dot next to "AWS" turns green
- Click "AWS" to open the console


STEP 1: Create  Your Server
1. Once inside AWS, click the search bar at the top
2. Type "EC2" and click it
3. Click the orange "Launch Instance" button
4. Fill in these details:
   - Name: nyc-taxi-fare-predictor
   - Operating System: Ubuntu 
   - Instance type: t2.micro
5. Create a Key Pair (this is like a password to access your server):
   - Click "Create new key pair"
   - Name it: taxi-key
   - Click "Create key pair"
   - A file called taxi-key.pem will download automatically
   - Save this file somewhere safe on your computer
   - You cannot download it again
6. Click "Launch Instance"
7. Wait 2-3 minutes for your server to start


STEP 2: Security Settings (Correct Ports)
1. Go to your EC2 dashboard
2. Click on your server name
3. Click the "Security" tab at the bottom
4. Click on the blue link under "Security groups"
5. Click "Edit inbound rules"
6. Add these three rules:

| Port  | Protocol  | Source  | Purpose  |
|-------|-----------|---------|----------|
| 22    | SSH       | My IP   | Connect to server |
| 8000  | TCP       | Anywhere| FastAPI backend access|
| 8501  | TCP       | Anywhere| Streamlit frontend access|

7. Click "Save rules"


STEP 3: Connect to your EC2 Instance
Open a terminal in VSCode and run the following:
ssh -i "taxi-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP

Where to find YOUR_EC2_PUBLIC_IP:
- Go to EC2 dashboard
- Click your server
- Look for "Public IPv4 address" 
- Copy that number (looks like: 54.123.456.789)

In case you get a permission error on a Windows machine, run:
icacls "taxi-key.pem" /inheritance:r /grant:r "%username%:R"

Then try the ssh command from above again. 


STEP 4: Install Docker and Git on the server (Tools Needed)
Run the following commands one at a time:

- Update the list of available software:

sudo apt-get update
- Install Git (downloads your code from GitHub):

sudo apt-get install -y git
- Install Docker (runs your containers):

sudo apt-get install -y docker.io
- Turn Docker on:

sudo systemctl start docker
sudo systemctl enable docker
- Grant yourself permission to use Docker without having to type 'sudo'

sudo usermod -aG docker ubuntu
- Log out so everything takes effect

exit
- Log back in

ssh -i "taxi-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP


STEP 5: Download the Project
- Dowload the code from GitHub onto the server: 

git clone https://github.com/gonzalesrios/final_project_nyc_taxi.git
- Go into the project folder:

cd final_project_nyc_taxi


STEP 6: Generate the Model File
The model file is not included in the repository  because it is too large.
You need to create it on the server. This takes 5-15 minutes. Make sure you do not close the terminal while this is running. 

- Install the required libraries:

pip install -r requirements.txt
- Train and save the model:

python train_model.py

                                                        
STEP 7: Set Up Docker Networking
- Create a shared folder for logs:

docker volume create my_volume
- Create a network so both containers can talk to each other:

docker network create my_network


STEP 8: Build Both Docker Images (Build App)
- Build the FastAPI:

docker build -t api_image ./api
- Build the Streamlit
docker build -t monitor_image ./monitoring


STEP 9: Run Both Containers (Start App)
- Start the FastAPI backend:

docker run -d --name api_container --network my_network -p 8000:8000 api_image
- Start the Streamlit frontend:

docker run -d --name monitor_container --network my_network -p 8501:8501 monitor_image


STEP 10:  Access Your Running App (Visit App)
Once both containers are running open your browser and go to:
- Streamlit Frontend (User Interface): 
`http://YOUR_EC2_PUBLIC_IP:8501`
- FastAPI Backend: `http://YOUR_EC2_PUBLIC_IP:8000`
- API Documentation: `http://YOUR_EC2_PUBLIC_IP:8000/docs`

Replace `YOUR_EC2_PUBLIC_IP` with your actual EC2 public IP address.

STEP 11: Stop Everything When Done
docker stop api_container monitor_container

docker rm api_container monitor_container

docker rmi api_image monitor_image

docker network rm my_network

docker volume rm my_volume


### Notes
-This project was built using AWS Academy Learner Lab through school.
- taxi_fare_model.pkl not included, its too large for GitHub
- Parquet data file not included, its too large for GitHub
  Download January 2026 Yellow Taxi data from NYC TLC website
- A .gitattributes file was not included but is recommended
  in real production projects
- AI assistance (Claude by Anthropic) was used during
  development for learning and confirming the use of new methods, all used appropriately and responsibly.
- Algorithm changed from RandomForestRegressor to 
  HistGradientBoostingRegressor to reduce model file size
  from 21GB to 0.35MB so it may be uploaded to GitHub.

-- Dev Branch --