#This file will: Load and Clean the data, Train the model, and Save the model.

import pandas as pd
import joblib
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

#LOAD THE DATA - STEP 1
print("Loading data...")
df = pd.read_parquet('yellow_tripdata_2026-01.parquet')
      #REMEMBER to use pd.read_parquet() to load the parquet file
print(f"Data loaded successfully. {df.shape[0]} rows and {df.shape[1]} columns.")


#-----------------------
#CLEAN THE DATA - STEP 2
# REMOVING rows where:
# 1. fare_amount <= 0 (negative or zero fares are impossible)
# 2. trip_distance <= 0 (zero or negative distance makes no sense)
# 3. passenger_count <= 0 (no passengers = not a real ride)
# 4. fare_amount > 200 (extreme outlier - real NYC rides rarely exceed $200)
# 5. trip_distance > 100 (extreme outlier - 269,097 miles is clearly wrong)

# FOR MISSING VALUES:
# passenger_count has 1M missing values (QUITE A LOT OF DATA)
# We will fill with median (middle value) instead of removing
# all the rows with missing passenger_count, because that would remove too much data
original_size = df.shape[0]
print("Before cleaning:", df.shape)

# Remove impossible values
df = df[df["fare_amount"] > 0]
df = df[df["fare_amount"] < 200]
df = df[df["trip_distance"] > 0]
df = df[df["trip_distance"] < 100]

# Fill missing passenger count with median
# We use median (middle value) instead of mean (average)
# because median is less affected by extreme values
df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)
#Remove rides with zero passengers
df = df[df["passenger_count"] > 0]

#Calculate and Show Results
rows_removed = original_size - df.shape[0]
rows_remaining = df.shape[0]
percentage_kept = (rows_remaining / original_size) * 100

print("After cleaning:", df.shape)
print(f"Rows removed: {rows_removed:,}")
print(f"Rows remaining: {rows_remaining:,}")
#tell me percentage of rows we kept/removed
print(f"Percentage of data kept: {percentage_kept:.2f}%")
print(f"Percentage of data removed: {100 - percentage_kept:.2f}%")


#----------------------------
# FEATURE ENGINEERING - STEP 3
#Extract time features from pickup datetime
#The model won't be able to read datetime objects like, "2026-01-01 00:00:05" 

print("\n" + "="*50)
print("FEATURE ENGINEERING")
#Extract hour of the day from pickup datetime (0-23)
#Rush hour in NYC is around 8am, 5-6pm, which I assume would have more traffic
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
print(f"Pickup hour feature created: range {df['pickup_hour'].min()} - {df['pickup_hour'].max()}")
print(f"Average pickup hour: {df['pickup_hour'].mean():.2f}")

#Extract day of the week feature (0-6) where 0=Monday, 6=Sunday
#0= Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
print(f"Pickup day of week feature created: range {df['pickup_dayofweek'].min()} - {df['pickup_dayofweek'].max()}")
print(f"Average pickup day of week: {df['pickup_dayofweek'].mean():.2f}")

#Extract in case its useful, the month of the year (1-12), weekend
df['pickup_weekend'] = df['pickup_dayofweek'].isin([5, 6]).astype(int)
print(f"Pickup weekend feature created: range {df['pickup_weekend'].min()} - {df['pickup_weekend'].max()}")
df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
print(f"Pickup month feature created: range {df['pickup_month'].min()} - {df['pickup_month'].max()}")


#---------------------------
#FEATURES AND TARGET - STEP 4
#Define the features and target variable for the model
#Features are the input and the target is the output we want to predict (fare amount in this case)
features = [
    'trip_distance',
    'passenger_count',
    'PULocationID',
    'DOLocationID',
    'pickup_hour',
    'pickup_dayofweek',
    'pickup_weekend', #feature i added above
    'pickup_month' #feature i added above
]

#X features, y target (output)
X = df[features]
y = df['fare_amount']

print(f"\nFeatures: {features}")
print(f"Target: fare_amount")
print(f"X shape: {X.shape}, y shape: {y.shape}")


#-------------------------------------------
#SPLIT DATA INTO TRAIN AND TEST SETS - STEP 5
#We will split the data into training and testing sets
#TESTING on training data means cheating, because the model has already seen the data and learned from it
#80% = Training, 20% = Testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set: {X_train.shape[0]} rows, percentage: {X_train.shape[0]/X.shape[0]*100:.2f}%")
print(f"\nTesting set: {X_test.shape[0]} rows, percentage: {X_test.shape[0]/X.shape[0]*100:.2f}%")


#-------------------------
#TRAIN THE MODEL - STEP 6
#RandomForestRegressor is good to help make many decisions and average them out to make a final decision
#We will switch to HistGradientBoostingRegressor for smaller model size and similar accuracy

"""
We will use 50 trees in the forest and then take the average of their predictions
n_estimators = number of trees in the forest, more trees = better performance but slower training
I tested 100 trees = 21GB file (too large for GitHub/Docker)
I tested 50 trees for accuracy and file size balance leading to better training speed and smaller model file size.
    50 trees = 10,792 MB (10GB) still large for GitHub/Docker
I tested 25 trees = 5,396 MB (5GB) which is a good balance but slightly larger than I want for GitHub/Docker
I tested 20 trees = 4,317 MB (4GB) which seems good but GitHub max is 50MG and Blocked at 100MG

Finding the best hyperparameters is called hyperparameter tuning. 
From the MachineLearning course I learned that I should use GridSearchCV to find the 
best hyperparameters, n_estimators, max_depth, min_samples_split, etc.

PROBLEM: Model file size must be under 100MB for GitHub and practical for Docker deployment
I tried n_estimators: 100, 50, 25, 20 and all were exceding 100MG limit for GitHub and Docker deployment. 

DECISION:
Originally used RandomForestRegressor but produced files of 4-21GB
model = RandomForestRegressor(n_estimators=20, 
                              n_jobs=-1, 
                              random_state=42)

Switched to HistGradientBoostingRegressor because:
- Designed for large datasets and faster training
- Produces much smaller model files
- Similar prediction accuracy
- Bins data into groups to reduce memory usaage
- Similar or better performance as RandomForestRegressor 
- Advised by Claude AI after telling it my problem of how large my files sizes were
https://stackoverflow.com/questions/53437426/optimize-random-forest-regressor-due-to-computational-limits
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- Responsible AI usage/acceptance was taken into account 

Final Results from testing HistGradientBoostingRegressor:
- File is 0.35 MB (350KB) which is very small and practical for GitHub and Docker deployment

"""
print("\n" + "="*50)
print("TRAINING MODEL")
model = HistGradientBoostingRegressor(max_iter=100, 
                                      random_state=42,
                                      max_depth=10,
                                      learning_rate=0.1,
                                      early_stopping=True)
model.fit(X_train, y_train) #model learns from this data


#Verify the model is trained by checking the training score (R2 score on training data)
#TRAINING METRICS:
r2_train = model.score(X_train, y_train)
train_predictions = model.predict(X_train)
mae_train = mean_absolute_error(y_train, train_predictions)
mse_train = mean_squared_error(y_train, train_predictions)
rmse_train = np.sqrt(mse_train)

print(f"Training Mean Absolute Error (MAE): {mae_train}")
print(f"Training Mean Squared Error (MSE): {mse_train}")
print(f"Training Root Mean Squared Error (RMSE): {rmse_train}")
print(f"Training R2 Score: {r2_train}")

#--------------------------
#EVALUATE THE MODEL - STEP 7
#Model Evaulation : Training = studying = X_train, y_train; Evaluation = taking the test = X_test, y_test
#Two metrics we will use to evaluate the model are:
#1. Mean Absolute Error (MAE) = average of the absolute errors between predicted and actual values
#2. Mean Squared Error (MSE) = average of the squared errors between predicted and actual values
#3.R2 Score = how well the model explains the variance in the data (1 = perfect, 0 = no better than mean, negative = worse than mean)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n" + "="*50)
print("EVALUATING MODEL PERFORMANCE")

#Make predictions on the test set
#Use X_test to predict fare_amount, and compare to actual y_test (real fares)
y_pred = model.predict(X_test)

#Calculate the metrics; avg dollar amount off, avg squared dollar amount off, and how well the model explains the variance in the data
mae_test = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test) #Root Mean Squared Error
r2_test = r2_score(y_test, y_pred)

#Print the metrics
print(f"Mean Absolute Error (MAE): {mae_test:.4f}")
print(f"On average, the model is off by ${mae_test:.2f} per ride of the actual fare amount.")
print()
print(f"Mean Squared Error (MSE): {mse_test:.4f}")
print(f"On average, the model is off by ${mse_test:.2f} squared per ride of the actual fare amount.")
print()
print(f"R2 Score: {r2_test}")
print(f"The model explains {r2_test*100:.2f}% of fare price variation") 

#Nice Comparison Table of Test vs Train Predictions
comparison_table = pd.DataFrame({
    'Metric': ['Mean Absolute Error (MAE)($)', 'Mean Squared Error (MSE)', 'RMSE ($)', 'R2 Score'],
    'Training Data': [f'{mae_train:.4f}', 
                      f'{mse_train:.2f}', 
                      f'{rmse_train:.4f}',
                      f'{r2_train:.4f}'],
    'Test Data': [f'{mae_test:.4f}',
                  f'{mse_test:.3f}',
                  f'{rmse_test:.3f}',
                  f'{r2_test:.4f}'],
    'Difference': [f'${abs(mae_train - mae_test):.2f}',
                   f'${abs(mse_train - mse_test):.2f}',
                   f'${abs(rmse_train - rmse_test):.2f}',
                   f'{abs(r2_train - r2_test):.4f}']
})

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("\n" + "="*60)
print(comparison_table.to_string(index=False))

# Check for overfitting
r2_difference = r2_train - r2_test
print("\n" + "="*60)
print("OVERFITTING CHECK ON R2 SCORE")
print(f"R2 difference (train - test): {r2_difference:.4f}")
if r2_difference > 0.1: #overfitting
    print("WARNING: Possible overfitting detected!")
    print("Training R2 much higher than Test R2")
    print("Model memorized training data instead of learning patterns")
    print("Consider reducing n_estimators or adding more training data")

elif r2_difference == 0.1: #borderline overfitting
    print("BORDERLINE: Right at the overfitting threshold")
    print("Model is acceptable but worth monitoring")
    print("Consider running with more data to confirm")

else:
    # r2_difference < 0.1 #healthy model, no overfitting
    print("No significant overfitting detected!")
    print("Model generalizes well to new unseen data")
    print("Training and test performance are close enough")
    
"""
R2 SCORE = CLOSE TO 1.0 IS = GOOD, MODEL EXPLAINS FARE PRICE VARIANCE WELL
MAE = UNDER $5 = GOOD, AVERAGE DOLLAR AMOUNT OFF PER RIDE OF THE ACTUAL FARE AMOUNT IS LOW
MSE = LOW = GOOD, PENALIZES LARGER ERRORS MORE THAN SMALLER ERRORS, SO LOW MSE = GOOD
TRAIN VS TEST = CLOSE TO EACH OTHER = GOOD, ~0.1 DIFFERENCE = GOOD, >0.1 DIFFERENCE = POSSIBLE OVERFITTING, <0.1 DIFFERENCE = GOOD, MODEL GENERALIZES WELL TO NEW DATA

MAE = Mean Absolute Error = average of the absolute errors between predicted and actual values
MSE = Mean Squared Error = average of the squared errors between predicted and actual values
R2 Score = how well the model explains the variance in the data (1 = perfect, 0 = no better than mean, negative = worse than mean)
RMSE = Root Mean Square Error = square root of MSE, back into original units (dollars), easier to interpret than MSE


"""


#-----------------------
#SAVE THE MODEL - STEP 8
#We will save the model to a file using joblib, so we can load it later
#Putting the model in a container/folder to take it to production, so we can use it in an API or web app
os.makedirs("api", exist_ok=True) #create api folder if it doesn't exist
model_path = "api/taxi_fare_model.pkl"
joblib.dump(model, model_path) # save the model to a file

#Show where the model is saved
print("\n" + "="*50)
print("MODEL SAVED")
print(f"Model saved to {model_path}")
print(f"Model file size: {os.path.getsize(model_path)/1024/1024:.2f} MB")
print("Training complete. Run `api/main.py` to serve predictions via FastAPI.")

#The FastAPI will load this  model file to make predictions 
#There's no need to retrain the model now or everytime we want to make predictions, UNLESS we get new data or want to improve the model. 
#In that case, we would retrain the model and save it again.