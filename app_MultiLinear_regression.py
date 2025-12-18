import streamlit as st  # type: ignore
import seaborn as sns  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore
import pandas as pd  # type: ignore

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Multiple Linear Regression",
    page_icon="📈",
    layout="centered"
)
# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")
# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h1>📈 Multiple Linear Regression</h1>
    <p>
        Predict <b>Tip Amount</b> from <b>Total Bill</b>, <b>Party Size</b>, and <b>Day of the Week</b>
        using a Multiple Linear Regression model.
    </p>
</div>
""", unsafe_allow_html=True)
# --------------------------------------------------

# --------------------------------------------------
# Prepare Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

X = df[["total_bill", "size"]]   # features
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# --------------------------------------------------
# Train Model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.markdown("""
<div class="card">      
    <h2>📋 Model Evaluation</h2>
    <ul>
        <li><b>Mean Squared Error (MSE):</b> {:.2f}</li>
        <li><b>Mean Absolute Error (MAE):</b> {:.2f}</li>
        <li><b>R² Score:</b> {:.2f}</li>
    </ul>
</div>
""".format(mse, mae, r2), unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h2>📊 Data Visualization</h2>
    <p>Scatter plot of Total Bill vs Tip with Regression Line.</p>
</div>
""", unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6, label="Actual Data")
ax.plot(
    df["total_bill"],
    model.predict(scaler.transform(df[["total_bill", "size"]])),
    color="red",
    label="Regression Line"
)   
ax.set_xlabel("Total Bill") 
ax.set_ylabel("Tip")
ax.legend()
st.pyplot(fig)


# --------------------------------------------------
# User Input for Prediction
# --------------------------------------------------
st.markdown("""
<div class="card">
    <h2>🧮 Make a Prediction</h2
    <p>Enter the details below to predict the tip amount.</p>
</div>
""", unsafe_allow_html=True)
bill_amount = st.slider(
    "Enter Total Bill ($)",
    min_value=float(df["total_bill"].min()),
    max_value=float(df["total_bill"].max()),
    value=30.0,
    step=1.0
)

party_size = st.slider(
    "Enter Party Size",
    min_value=int(df["size"].min()),
    max_value=int(df["size"].max()),
    value=2,
    step=1
)


if st.button("Predict Tip Amount"): 
    input_data = np.array([[bill_amount, party_size]])
    input_data_scaled = scaler.transform(input_data)
    predicted_tip = model.predict(input_data_scaled)
    st.success(f"Predicted Tip Amount: ${predicted_tip[0]:.2f}")

# --------------------------------------------------
# The scatter plot shows the actual data points (in blue) and the regression line (in
# red). The regression line represents the predicted tip amounts based on the total bill and party size.
# The visualization helps to understand how well the regression line fits the data.
# Input: total_bill, size; Output: tip
