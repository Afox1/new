import streamlit as st
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CNG ROI App", layout="centered")
st.title("🔍 CNG Savings & Investment Return App")

# Input fields
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, step=0.1)
distance = st.number_input("Distance Travelled (km)", min_value=1.0, max_value=10000.0, step=1.0)
install_cost = st.number_input("Cost of Installation (#)", min_value=0.0, step=100.0)

# Predict and display result
if engine_size > 0 and distance > 0:
    # Scale inputs with predefined mean and scale
    scaler = StandardScaler()
    scaler.mean_ = np.array([2.038, 207.679])
    scaler.scale_ = np.array([0.688, 111.230])
    inputs = np.array([[engine_size, distance]])
    inputs_scaled = scaler.transform(inputs)

    # Dummy model structure
    model = MLPRegressor(hidden_layer_sizes=(13, 13, 13), solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(np.zeros((2, 2)), [0, 0])  # dummy fit

    predicted_savings = 35000 + (distance * 5) + (engine_size * 1000)
    st.success(f"Predicted Savings: ₦{predicted_savings:,.2f}")

    profit = predicted_savings - install_cost
    if profit >= 0:
        st.info(f"🎉 You are now making a profit of ₦{profit:,.2f}")
    else:
        st.warning(f"⏳ You need ₦{abs(profit):,.2f} more to break even.")
else:
    st.write("👉 Please enter valid engine size and distance to continue.")
