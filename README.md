# Forecasting Economic Gap of Developing Countries

This Streamlit app demonstrates time series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model. The application focuses on predicting the economic gap (GDP difference) of developing countries based on historical data.

## Setup

Before running the app, ensure you have the required libraries installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```
## Usage
### Load Dataset:
The app loads a dataset from a CSV file named cleaned_dataset_with_incomegroup.csv. The data is sorted by the year and set as the index.

### Sidebar Configuration:
The sidebar allows you to configure the ARIMA model by selecting a country and adjusting the electrification level. These parameters influence the training of the ARIMA model.

ARIMA Model Building:
The ARIMA model is built based on the selected country and electrification level. The data is split into training and testing sets, and the model is fitted to the training data.

Model Evaluation:
The Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) of the ARIMA model are displayed to assess the model’s performance on the test set.

Forecasting:
The app extends the time index to forecast the economic gap for the next 20 years. The predicted values are visualized using a dashed orange line on the economic gap plot.

How to Run
To run the app, execute the following command:

streamlit run your_script_name.py

Replace your_script_name.py with the filename of your script containing the Streamlit app.

Dependencies
streamlit
pytz
pandas
numpy
matplotlib
statsmodels
scikit-learn
