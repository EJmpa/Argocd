# Library importation
import streamlit as st
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit's cache mechanism to load datasets efficiently
@st.cache_resource
def load_data(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(by='year').set_index('year')
    df.index = pd.to_datetime(df.index, format='%Y')
    # df.index = df.index.strftime('%Y')
    return df

# Load your dataset
data = load_data("./data/cleaned_dataset_with_incomegroup.csv")
print(data.head())

# Filter data for Low-Income Countries
developing_countries = data[data['IncomeGroup'] != 'High income']

# Sidebar for ARIMA model configuration
st.sidebar.header('ARIMA Model Configuration')

# Sidebar for selecting country and elecrate_total range
selected_country = st.sidebar.selectbox('Select a Country', developing_countries['countryname'].unique())
selected_elecrate_total = st.sidebar.slider('Electrification Level', 
                                           min_value=developing_countries['elecrate_total'].min(), 
                                           max_value=developing_countries['elecrate_total'].max(),
                                           value=developing_countries['elecrate_total'].median())
# Function to fit ARIMA model
def fit_arima_model(data):
    model = ARIMA(data, order=(16, 0, 3))
    fitted_model = model.fit()
    return fitted_model

# Function to make ARIMA predictions
def make_arima_predictions(model, start, end):
    predictions = model.predict(start=start, end=end, typ='levels')
    return predictions

# Function to make ARIMA forecast
def make_arima_forecast(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast

# Function to evaluate ARIMA model
def evaluate_arima_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return mae, rmse

# ARIMA Model Building
target_column = "economicgap (GDP difference)"

# Filter data for the selected country and elecrate_total range
selected_country_data = developing_countries[(developing_countries['countryname'] == selected_country) & 
                                             (developing_countries['elecrate_total'] == selected_elecrate_total)]


# Split the data into train and test
train_size = int(len(developing_countries) * 0.8)
train, test = developing_countries[[target_column, 'elecrate_total']][:train_size], developing_countries[[target_column, 'elecrate_total']][train_size:]
# Check if train DataFrame is empty

if train.empty:
    st.error('Training data is empty.')
else:
    # Fit ARIMA model
    arima_model = fit_arima_model(train[target_column])

    # Make predictions
    predictions = make_arima_predictions(arima_model, start=len(train), end=len(train) + len(test) - 1)

    # Evaluate the model
    mae = evaluate_arima_model(test[target_column], predictions)[0]
    rmse = evaluate_arima_model(test[target_column], predictions)[1]

# Streamlit app
st.title('Forecasting Economic Gap of Developing Countries')
st.write(f'Mean Absolute Error (MAE): {mae}')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')


# Extend the time index for forecasting
forecast_years = 20
forecast_index = pd.date_range(test.index[-1], periods=forecast_years * 12 + 1, freq='M')[1:]

# Make predictions for the forecast period
# forecast_start = len(train) + len(test)
# forecast_end = forecast_start + len(forecast_index) - 1
# forecast = make_arima_predictions(arima_model, start=forecast_start, end=forecast_end)
forecast = make_arima_forecast(arima_model, steps=forecast_years*12)

# Function to format y-axis values
def billions(x, pos):
    'The two args are the value and tick position'
    return '%1.0f Billion' % (x * 1e-9)

formatter = FuncFormatter(billions)

# Plot actual vs predicted values
fig, ax = plt.subplots(figsize=(10, 6))
ax.yaxis.set_major_formatter(formatter)
# ax.plot(test.index, test[target_column], label='Actual')
# ax.plot(test.index, predictions, label='Predicted', color='blue')
ax.plot(forecast_index, forecast, label='Forecast', linestyle='dashed', color='orange')
ax.set_title('Economic Gap Prediction')
ax.set_xlabel('Year')
ax.set_ylabel('Economic Gap (GDP Differrence) $')
ax.legend()
st.pyplot(fig)
