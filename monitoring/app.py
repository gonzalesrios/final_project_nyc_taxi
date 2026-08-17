#This is the Streamlit frontend for the monitoring application. It allows users to visualize and monitor the performance of their machine learning models in real-time.
#User will enter trip details and the application will predict the fare amount based on the input features. 
#Will call FastAPI backend to get the prediction and display it on the frontend.

import streamlit as st
import requests

#FastAPI backend URL
#Sends prediction request to the FastAPI backend and returns the predicted fare amount.
API_URL = "http://localhost:8000/predict"


#----------
# PAGE SETUP 
st.set_page_config(
    page_title="NYC Taxi Fare Predictor",
    page_icon="🚕",
    layout="centered"
)


#---------------
# Header Section
st.markdown(
    """
    <h1 style='text-align: center; color: #FFD700;'>
        NYC Taxi Fare Predictor
    </h1>
    <p style='text-align: center; color: darkblue;'>
       Trained using 3.5 million NYC taxi trips from January 2026
    </p>
    """,
    unsafe_allow_html=True,
)


#---------------------
# Instructions Section
st.subheader("How to use the application:")
st.markdown("""
1. Select your "pickup neighborhood" and "dropoff neighborhood" from the dropdown menus.
2. Enter how 'far' (distance) your trip is in miles.
3. Choose the 'date and time' of your trip.
4. Click the "Predict Fare" button to get an estimated fare amount for your trip.
5. Remember, this is an estimate only and actual fares may vary based on traffic, route, and other factors.
""")

st.warning("""
Please avoid:
- Entering negative numbers
- Entering zero for distance or passengers
- Entering hours outside 0-23 range
""")

st.divider() #to add a horizontal line

#NYC ZONES DICTIONARY
# Format: "Zone [number] - [Neighborhood] ([Borough])"
# User sees both name AND number - helpful for tourists and locals alike
nyc_zones = {
    "Zone 1 - Newark Airport": 1,
    "Zone 4 - Alphabet City (Manhattan)": 4,
    "Zone 7 - Astoria (Queens)": 7,
    "Zone 12 - Battery Park (Manhattan)": 12,
    "Zone 13 - Battery Park City (Manhattan)": 13,
    "Zone 14 - Bay Ridge (Brooklyn)": 14,
    "Zone 17 - Bedford (Brooklyn)": 17,
    "Zone 25 - Bloomingdale (Manhattan)": 25,
    "Zone 26 - Boerum Hill (Brooklyn)": 26,
    "Zone 35 - Brooklyn Heights (Brooklyn)": 35,
    "Zone 36 - Brownsville (Brooklyn)": 36,
    "Zone 37 - Bushwick (Brooklyn)": 37,
    "Zone 40 - Carroll Gardens (Brooklyn)": 40,
    "Zone 42 - Central Harlem (Manhattan)": 42,
    "Zone 43 - Central Park (Manhattan)": 43,
    "Zone 45 - Chinatown (Manhattan)": 45,
    "Zone 48 - Clinton Hill/Hell's Kitchen (Manhattan)": 48,
    "Zone 49 - Clinton Hill (Brooklyn)": 49,
    "Zone 52 - Cobble Hill (Brooklyn)": 52,
    "Zone 55 - Coney Island (Brooklyn)": 55,
    "Zone 57 - Crown Heights North (Brooklyn)": 57,
    "Zone 58 - Crown Heights South (Brooklyn)": 58,
    "Zone 61 - Downtown Brooklyn": 61,
    "Zone 66 - DUMBO (Brooklyn)": 66,
    "Zone 74 - East Harlem (Manhattan)": 74,
    "Zone 77 - East New York (Brooklyn)": 77,
    "Zone 79 - East Village (Manhattan)": 79,
    "Zone 85 - Flatbush (Brooklyn)": 85,
    "Zone 87 - Flushing (Queens)": 87,
    "Zone 90 - Forest Hills (Queens)": 90,
    "Zone 92 - Fort Greene (Brooklyn)": 92,
    "Zone 100 - Garment District (Manhattan)": 100,
    "Zone 103 - Gowanus (Brooklyn)": 103,
    "Zone 104 - Gramercy (Manhattan)": 104,
    "Zone 113 - Greenwich Village (Manhattan)": 113,
    "Zone 116 - Hamilton Heights (Manhattan)": 116,
    "Zone 117 - Harlem (Manhattan)": 117,
    "Zone 127 - Inwood (Manhattan)": 127,
    "Zone 129 - Jackson Heights (Queens)": 129,
    "Zone 132 - JFK Airport (Queens)": 132,
    "Zone 134 - Kensington (Brooklyn)": 134,
    "Zone 138 - LaGuardia Airport (Queens)": 138,
    "Zone 140 - Lenox Hill East (Manhattan)": 140,
    "Zone 141 - Lenox Hill West (Manhattan)": 141,
    "Zone 142 - Lincoln Square East (Manhattan)": 142,
    "Zone 148 - Little Italy/NoLiTa (Manhattan)": 148,
    "Zone 150 - Long Island City (Queens)": 150,
    "Zone 151 - Lower East Side (Manhattan)": 151,
    "Zone 161 - Midtown Center (Manhattan)": 161,
    "Zone 162 - Midtown East (Manhattan)": 162,
    "Zone 163 - Midtown North (Manhattan)": 163,
    "Zone 164 - Midtown South (Manhattan)": 164,
    "Zone 166 - Morningside Heights (Manhattan)": 166,
    "Zone 170 - Murray Hill (Manhattan)": 170,
    "Zone 172 - Park Slope (Brooklyn)": 172,
    "Zone 186 - Penn Station/Madison Sq West (Manhattan)": 186,
    "Zone 190 - Prospect Heights (Brooklyn)": 190,
    "Zone 202 - Red Hook (Brooklyn)": 202,
    "Zone 203 - Rego Park (Queens)": 203,
    "Zone 209 - Riverdale (Bronx)": 209,
    "Zone 213 - Rockaway Park (Queens)": 213,
    "Zone 214 - Roosevelt Island (Manhattan)": 214,
    "Zone 222 - Sheepshead Bay (Brooklyn)": 222,
    "Zone 224 - SoHo (Manhattan)": 224,
    "Zone 228 - South Beach (Staten Island)": 228,
    "Zone 230 - Times Square/Theatre District (Manhattan)": 230,
    "Zone 234 - Stapleton (Staten Island)": 234,
    "Zone 236 - Stuyvesant Heights (Brooklyn)": 236,
    "Zone 238 - Sunnyside (Queens)": 238,
    "Zone 239 - Sunset Park (Brooklyn)": 239,
    "Zone 243 - Two Bridges/Seaport (Manhattan)": 243,
    "Zone 244 - Union Square (Manhattan)": 244,
    "Zone 245 - Upper East Side North (Manhattan)": 245,
    "Zone 246 - Upper East Side South (Manhattan)": 246,
    "Zone 247 - Upper West Side North (Manhattan)": 247,
    "Zone 248 - Upper West Side South (Manhattan)": 248,
    "Zone 249 - Washington Heights (Manhattan)": 249,
    "Zone 250 - West Village (Manhattan)": 250,
    "Zone 261 - Williamsburg (Brooklyn)": 261,
    "Zone 265 - Woodside (Queens)": 265,
}

st.divider() #to add a horizontal line


#---------------------
#INPUT SECTION FOR USER
st.subheader("Enter Your Trip Details:")
#Have two columns for input fields
col1, col2 = st.columns(2)

with col1:
    trip_distance = st.number_input("Trip Distance (miles)", 
                                    min_value=0.1, 
                                    max_value=100.0,
                                    value=2.0,
                                    step=0.1)
    passenger_count = st.number_input("Number of Passengers",
                                        min_value=1, 
                                        max_value=8,
                                        value=1, 
                                        step=1)
    pickup_month = st.selectbox("Pickup Month",
                                options=list(range(1, 13)),
                                format_func=lambda x: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][x-1], )

with col2:
    pickup_hour = st.slider("Pickup Hour (0-23)",
                            min_value=0,
                            max_value=23,
                            value=12,
                            step=1)
    pickup_dayofweek = st.selectbox("Day of Week",
                                        options=list(range(0, 7)),
                                        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])

#Outside of col1 and col2 (full width dropdowns)
pickup_zone_name = st.selectbox(
    "Where are you getting picked up?",
    options=sorted(nyc_zones.keys()),
    index=list(sorted(nyc_zones.keys())).index("Zone 132 - JFK Airport (Queens)") 
        #default pickup for tourists/locals 
    
)

dropoff_zone_name = st.selectbox(
    "Where are you getting dropped off?",
    options=sorted(nyc_zones.keys()),
    index=list(sorted(nyc_zones.keys())).index("Zone 161 - Midtown Center (Manhattan)")
        #dropoff default bc its a common destination for tourists and locals alike
)
    
# Convert selected name to zone ID - user wont see this
PULocationID = nyc_zones[pickup_zone_name]
DOLocationID = nyc_zones[dropoff_zone_name]
    
#Calculate weekend 
pickup_weekend = 1 if pickup_dayofweek >=5 else 0
st.caption(f"Weekend: {'Yes' if pickup_weekend else 'No'}")
    
st.divider() #to add a horizontal line
  
  
#------------------    
#PREDICT FARE BUTTON
predict_button = st.button("Predict Fare",
                                type="primary",
                                use_container_width=True)


#--------------------------   
# PREDICTION RESULTs SECTION
if predict_button:
    # Prepare the input data for the API request
    input_data = {
        "trip_distance": trip_distance,
        "passenger_count": passenger_count,
        "pickup_month": pickup_month,
        "pickup_hour": pickup_hour,
        "pickup_dayofweek": pickup_dayofweek,
        "pickup_weekend": pickup_weekend,
        "PULocationID": PULocationID,
        "DOLocationID": DOLocationID,
    }

    response = None

    try:
        # Send POST request to FastAPI backend
        response = requests.post(API_URL, json=input_data, timeout=10)
        response.raise_for_status()
        prediction = response.json().get("predicted_fare")

        # Display the predicted fare amount
        st.success(f"Estimated Fare Amount: ${prediction:.2f}")


        #Trip summary for successful requests
        st.subheader("Your Trip Summary")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Distance", f"{trip_distance} miles")
            st.metric("Passengers", passenger_count)
            st.metric("From", pickup_zone_name)
        with col4:
            st.metric("To", dropoff_zone_name)
            st.metric("Pickup Time", f"{pickup_hour}:00")
            st.metric("Weekend", "Yes" if pickup_weekend else "No")

        st.caption("Note: This is an estimate. Actual fare may vary. Thank you.")
        
    except requests.exceptions.HTTPError:
        st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the prediction API!")

    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")