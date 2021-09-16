# pip install pandas numpy matplotlib streamlit pystan fbprophet cryptocmd plotly
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
from datetime import date, datetime
from cryptocmd import CmcScraper
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

st.title('The Future of CRO (Tech Sharing DEMO)')
st.markdown("The application predict the future value of CRO for any number of days into the future. It is built with Streamlit and Facebook Prophet prediction model. The original program is made by Michael Tuijp (https://medium.com/analytics-vidhya/predicting-cryptocurrency-prices-using-facebook-prophet-a1509415224f), here is just a simplify version just for demo.")

### Style setting----
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#D6EAF8,#D6EAF8);
    color: black;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
	"""
<style>
.big-font {
	fontWeight: bold;
    font-size:22px !important;
}
</style>
""", unsafe_allow_html=True)

### Sidebar----
st.sidebar.markdown("<p class='big-font'>Settings</font></p>", unsafe_allow_html=True)

selected_ticker = "CRO" #Can change into other crypto
period = int(st.sidebar.number_input('Number of days to predict:', min_value=0, max_value=1000000, value=365, step=1))
training_size = int(st.sidebar.number_input('Training set (%) size:', min_value=10, max_value=100, value=100, step=5)) / 100

### Get crypto market data----
@st.cache
def load_data(selected_ticker):
	init_scraper = CmcScraper(selected_ticker)
	df = init_scraper.get_dataframe()
	min_date = pd.to_datetime(min(df['Date']))
	max_date = pd.to_datetime(max(df['Date']))
	return min_date, max_date

data_load_state = st.sidebar.text('Loading data...')
min_date, max_date = load_data(selected_ticker)
data_load_state.text('Loading data... done!')
scraper = CmcScraper(selected_ticker)
data = scraper.get_dataframe()

st.subheader('Historical data from Coinmarketcap.com')
st.write(data.head())

### Plot data----
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
plot_raw_data()

### Prediction with Prophet
if st.button("Predict"):
	df_train = data[['Date','Close']]
	df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

	m = Prophet(
		changepoint_range=training_size, # 0.8
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality=False,
        seasonality_mode='multiplicative', # multiplicative/additive
        changepoint_prior_scale=0.05
		)

	for col in df_train.columns:
	    if col not in ["ds", "y"]:
	        m.add_regressor(col, mode="additive")
	
	m.fit(df_train)

	### Predict using the model
	future = m.make_future_dataframe(periods=period)
	forecast = m.predict(future)

	### Show and plot forecast
	st.subheader('Forecast data')
	st.write(forecast.head())
	    
	st.subheader(f'Forecast plot for {period} days')
	fig1 = plot_plotly(m, forecast)
	st.plotly_chart(fig1)