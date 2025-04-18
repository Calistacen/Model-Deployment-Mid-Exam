import pickle
import numpy as np
import streamlit as st

def predict_booking_status(input_data):
    with open('CL_ranfor_model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return prediction

def main():
    st.title("Hotel Booking Status Prediction")
    st.markdown("#### Use this app to predict whether your guest will possibly cancel or proceed with the booking they made before.")

    st.markdown("### Guest Information")
    adults = st.number_input("Number of Adults", 1, 4, step=1)
    children = st.number_input("Number of Children", 0, step=1)

    market_options = {
        "Online": 4,
        "Offline": 3,
        "Corporate": 2,
        "Complementary": 1,
        "Aviation": 0
    }
    market = st.selectbox("Type of Booking", list(market_options.keys()))
    market_value = market_options[market]

    st.markdown("### Stay Duration")
    weekday_nights = st.slider("Weekday Nights (Mon - Fri)", 0, 30)
    remaining = 30 - weekday_nights
    weekend_nights = st.slider("Weekend Nights", 0, remaining)

    st.markdown("### Parking & Requests")
    required_car_parking_space = int(st.checkbox("Guest need parking slot?"))
    special_requests = st.slider("Special Requests", 0, 5)

    st.markdown("### Booking Info")
    lead_time = st.number_input("How many days guest booked before check-in?", 0, step=1)
    year = st.selectbox("Arrival Year", [2017, 2018])

    month_options = {
        "January": 1, 
        "February": 2, 
        "March": 3, 
        "April": 4,
        "May": 5, 
        "June": 6, 
        "July": 7, 
        "August": 8,
        "September": 9, 
        "October": 10, 
        "November": 11, 
        "December": 12 
    }
    month = st.selectbox("Arrival Month", list(month_options.keys()))
    month_value = month_options[month]

    day = st.number_input("Arrival Day", min_value=1, max_value=31, step=1)
    avg_price = st.number_input("Avg Price per Room (Euro)", 0.0)
    room_type = st.radio("Type of Room Booked", [1, 2, 3, 4, 5, 6, 7])
    meal_options = {
        "Not Selected": 3,
        "Meal Plan 1": 0,
        "Meal Plan 2": 1,
        "Meal Plan 3": 2
    }
    meal = st.selectbox("Type of Meal Plan Choosen", list(meal_options.keys()))
    meal_value = meal_options[meal]

    st.markdown("### Guest History")
    repeated_guest = int(st.checkbox("Guest ever booked before?"))
    if repeated_guest:
        previous_booking = st.number_input("Number of previous booking made", min_value = 1)
        previous_cancelation = st.number_input("Guest cancelation before this booking", 0)
    else:
        previous_booking = 0
        previous_cancelation = 0
    
    
    input_data = [adults, children, weekend_nights, weekday_nights, meal_value, required_car_parking_space, room_type, lead_time, year, month_value, day, market_value, repeated_guest, previous_cancelation, previous_booking, avg_price, special_requests]

    
    if st.button("Predict"):
        with st.spinner("Processing your booking... Please wait."):
            prediction = predict_booking_status(input_data)
            
        if prediction == 0:
            st.error("❌ The guest will most likely cancel this booking.")
        elif prediction == 1:
            st.success("✅ The guest will most likely proceed with this booking.")
        else:
            st.warning("⚠️ Unable to determine the booking status.")

st.markdown("""
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        left: 10px;
        color: white;
        background-color: rgba(0, 0, 0, 0.6); 
        padding: 8px 16px;
        font-size: 13px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 9999;
        font-family: 'Segoe UI', sans-serif;
        
    }
    </style>
    <div class="watermark">Made by Calista Lianardi - 2702325880</div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
