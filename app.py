# Library importation
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Streamlit's cache mechanism to load datasets efficiently
# @st.cache_resource
def load_data(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by='year').set_index('year')
    df.index = pd.to_datetime(df.index, format='%Y')
    return df

# Load your dataset
data = load_data("./data/cleaned_dataset_with_incomegroup.csv")

# Filter data for Low-Income Countries
developing_countries = data[data['IncomeGroup'] != 'High income']

# Sidebar for ARIMA model configuration
st.sidebar.header('ARIMA Model Configuration')
order_p = st.sidebar.slider('Order (p) for ARIMA', min_value=0, max_value=5, value=1)
order_d = st.sidebar.slider('Order (d) for ARIMA', min_value=0, max_value=5, value=1)
order_q = st.sidebar.slider('Order (q) for ARIMA', min_value=0, max_value=5, value=1)

# Sidebar for selecting country and elecrate_total range
selected_country = st.sidebar.selectbox('Select a Country', developing_countries['countryname'].unique())
selected_elecrate_total = st.sidebar.slider('Select elecrate_total', 
                                           min_value=developing_countries['elecrate_total'].min(), 
                                           max_value=developing_countries['elecrate_total'].max(),
                                           value=developing_countries['elecrate_total'].median())
# Function to fit ARIMA model
def fit_arima_model(data, order):
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

# Function to make ARIMA predictions
def make_arima_predictions(model, start, end):
    predictions = model.predict(start=start, end=end, typ='levels')
    return predictions

# Function to evaluate ARIMA model
def evaluate_arima_model(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return rmse

# ARIMA Model Building
target_column = "economicgap (GDP difference)"

# Filter data for the selected country and elecrate_total range
selected_country_data = developing_countries[(developing_countries['countryname'] == selected_country) & 
                                             (developing_countries['elecrate_total'] == selected_elecrate_total)]


# Split the data into train and test
train_size = int(len(selected_country_data) * 0.8)
train, test = selected_country_data[[target_column, 'elecrate_total']][:train_size], selected_country_data[[target_column, 'elecrate_total']][train_size:]

# Drop rows with missing values (if any)
train = train.dropna()
test = test.dropna()

# Fit ARIMA model
arima_order = (order_p, order_d, order_q)
arima_model = fit_arima_model(train[target_column], order=arima_order)

# Make predictions
predictions = make_arima_predictions(arima_model, start=len(train), end=len(train) + len(test) - 1)

# Evaluate the model
rmse = evaluate_arima_model(test[target_column], predictions)

# Streamlit app
st.title('ARIMA Time Series Model for Low-Income Countries')
st.write(f'ARIMA Model Order: {arima_order}')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot actual vs predicted values
fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(test.index, test[target_column], label='Actual')
ax.plot(test.index, predictions, label='Predicted', color='blue')
ax.set_title('Economic Gap Prediction')
ax.set_xlabel('Year')
ax.set_ylabel('Economic Gap (GDP Differrence)')
ax.legend()
st.pyplot(fig)
