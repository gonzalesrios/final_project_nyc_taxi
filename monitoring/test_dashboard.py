#Simple test to verify Streamlit dashboard was able to be imported without errors

#REFERENCES:
#https://discuss.streamlit.io/t/load-stress-testing-on-streamlit-web-app/63604
#https://docs.streamlit.io/develop/concepts/app-testing/get-started
#https://docs.streamlit.io/develop/api-reference/app-testing

from streamlit.testing.v1 import AppTest

def test_app_loads_without_errors():
    "Test that the app loads without crashing"
    at = AppTest.from_file("monitoring/app.py").run()
    assert not at.exception

def test_app_has_correct_title():
    "Test that the app title is correct"
    at = AppTest.from_file("monitoring/app.py").run()
    assert at.title[0].value == "NYC Taxi Fare Predictor"

def test_predict_button_exist():
    "Test the predict fare button exists"
    at = AppTest.from_file("monitoring/app.py").run()
    assert at.button[0].label == "Predict Fare"
    
def test_dropdowns_exist():
    "Test that pickup and dropoff dropdowns exist"
    at = AppTest.from_file("monitoring/app.py").run()
    assert len(at.selectbox) >= 2

