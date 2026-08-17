#This is a temporary script to explore the data and understand the structure of the dataset. It will help in identifying any missing values, data types, and basic statistics of the features.
#WE are using 'parquet' format NOT 'csv' format
#Read with pd.read_parquet() function not pd.read_csv() function


#REMEMBER STEPS: LOAD DATA, CLEAN DATA, FEATURE ENGINEERING, SELECT FEATURES/TARGET, SPLIT DATA, TRAIN MODEL, EVALUATE MODEL, SAVE MODEL

import pandas as pd
import numpy as np

#Load the dataset-taxi data
#Parquet files are more efficient than CSV files in terms of storage and speed. 
#They are also better suited for large datasets.
df = pd.read_parquet('yellow_tripdata_2026-01.parquet')

#Basic information about the dataset
print("Basic Information about the dataset:")
#number of rows and columns
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
#name of the columns
print(f"Column names: {df.columns.tolist()}")
#data types of the columns
print(f"Data types of the columns:\n{df.dtypes}")
print()
#view the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())
print()
#basic statistics of the dataset
print("Basic statistics of the dataset:")
print(df.describe(include='all'))

#From this we have learned:
#20 Columns in the dataset & 3,724,889 rows
#Column names: VendorID, 
# tpep_pickup_datetime, 
# tpep_dropoff_datetime, 
# passenger_count, 
# trip_distance, 
# RatecodeID, 
# store_and_fwd_flag, 
# PULocationID, 
# DOLocationID, 
# payment_type, 
# fare_amount, 
# extra, 
# mta_tax, 
# tip_amount, 
# tolls_amount, 
# improvement_surcharge, 
# total_amount, 
# congestion_surcharge, 
# airport_fee
# cbd_congestion_fee

#Important/Key Columns to focus on for analysis:
# tpep_pickup_datetime - when the ride started
# trip_distance - how far the ride was
# passenger_count - how many passengers were in the ride
# PULocationID - pickup location ID/zone
# DOLocationID - dropoff location ID/zone
# fare_amount - how much the ride cost - Target Variable
# total_amount - how much the ride cost including tip, tolls, and other fees - Similar to Target Variable
# tip_amount - how much the passenger tipped the driver - Unknown at Prediction Time

#TASK
#Given pickup location, dropoff location, distance, time of day and passenger count, predict the fare amount for a ride.    

#DATA CLEANING EXPLORATION
# Clean the data
#Remove null values, zero values, negative values, empty values, and outliers
# Before cleaning anything, we need to UNDERSTAND what we have

# STEP 1: Check for missing/null values
# Null means empty - no value recorded for that ride
print("="*50)
print("MISSING VALUES CHECK")
print("="*50)
print(df.isnull().sum())
# This shows how many empty values exist in each column
# If a column has many nulls, we need to decide:
# Option A: Remove those rows
# Option B: Fill them with average/median value

# STEP 2: Check for zero values in important columns
print("\n" + "="*50)
print("ZERO VALUES CHECK")
print("="*50)
print(f"Rides with zero fare: {(df['fare_amount'] == 0).sum()}")
print(f"Rides with zero distance: {(df['trip_distance'] == 0).sum()}")
print(f"Rides with zero passengers: {(df['passenger_count'] == 0).sum()}")
# Zero fare = free ride? Probably an error
# Zero distance = cab never moved? Probably an error
# Zero passengers = ghost ride? Definitely an error

# STEP 3: Check for negative values
print("\n" + "="*50)
print("NEGATIVE VALUES CHECK")
print("="*50)
print(f"Rides with negative fare: {(df['fare_amount'] < 0).sum()}")
print(f"Rides with negative distance: {(df['trip_distance'] < 0).sum()}")
# Negative fare makes no sense in real life
# Negative distance is physically impossible

# STEP 4: Check for extreme outliers
print("\n" + "="*50)
print("OUTLIERS CHECK")
print("="*50)
print(f"Max fare amount: ${df['fare_amount'].max()}")
print(f"Min fare amount: ${df['fare_amount'].min()}")
print(f"Average fare amount: ${df['fare_amount'].mean():.2f}")
print(f"Max trip distance: {df['trip_distance'].max()} miles")
print(f"Average trip distance: {df['trip_distance'].mean():.2f} miles")
# If max fare is $100,000 that's probably an error
# Real NYC taxi rides are usually ~$5 - $200


# DATA CLEANING AND VERIFICATION
# Now we clean and verify the data looks good after cleaning

print("Before cleaning:", df.shape)

# Remove impossible values
df = df[df["fare_amount"] > 0]
df = df[df["fare_amount"] < 200]
df = df[df["trip_distance"] > 0]
df = df[df["trip_distance"] < 100]

# Fill missing passenger count with median
df["passenger_count"] = df["passenger_count"].fillna(
    df["passenger_count"].median()
)
df = df[df["passenger_count"] > 0]

print("After cleaning:", df.shape)
print(f"Rows removed: {3724889 - df.shape[0]:,}")
print(f"Rows remaining: {df.shape[0]:,}")
Percentage_kept = (df.shape[0] / 3724889) * 100
print(f"Percentage of data kept: {Percentage_kept:.2f}%")

# Extract time features
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek

# FINAL VERIFICATION - each column separately
key_columns = [
    "fare_amount", "trip_distance", "passenger_count",
    "PULocationID", "DOLocationID", "pickup_hour", "pickup_dayofweek"
]

print("\n" + "="*60)
print("FINAL DATA SUMMARY AFTER CLEANING")
print("="*60)
for col in key_columns:
    print(f"\n{col}:")
    print(f"  Min:     {df[col].min():.2f}")
    print(f"  Max:     {df[col].max():.2f}")
    print(f"  Average: {df[col].mean():.2f}")
    print(f"  Missing: {df[col].isnull().sum()}")
    

#VERIFICATION/DATA EXPLORATION - STEP 3
#Want to check/verify that the data kept is reasonable and makes sense
#QUALITY CHECK FOR CLEANED DATA


#fare amount range check
print("\n" + "="*50)
print("FARE AMOUNT RANGE CHECK")
print(f"Max fare amount: ${df['fare_amount'].max()}")
print(f"Min fare amount: ${df['fare_amount'].min()}")

#average fare amount check
print(f"Average fare amount: ${df['fare_amount'].mean():.2f}")

#distance range check
#shortest to longest trip distance check
print("\n" + "="*50)
print("TRIP DISTANCE RANGE CHECK")
print(f"Max trip distance: {df['trip_distance'].max()}")
print(f"Min trip distance: {df['trip_distance'].min()}")

#average trip distance check
print(f"Average trip distance: {df['trip_distance'].mean():.2f} miles")

#passenger count range check
print("\n" + "="*50)
print("PASSENGER COUNT RANGE CHECK")
print(f"Max passenger count: {df['passenger_count'].max()}")
print(f"Min passenger count: {df['passenger_count'].min()}")

#passenger count average check
print(f"Average passenger count: {df['passenger_count'].mean():.2f}")   


# Create a clean summary tablble of all KEY COLUMNS
print("\n" + "="*50)
print("Final Data Summary - After Cleaning")

#Extract time features
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek

#Key Columns are:
key_columns = ['fare_amount', 'trip_distance', 'passenger_count', 'pickup_hour', 'pickup_dayofweek', 'PULocationID', 'DOLocationID']

#Show each column
for col in key_columns:
    print(f"\nColumn: {col}")
    print(df[col].describe().round(2))
    print(f"Missing values: {df[col].isnull().sum()}")
    

