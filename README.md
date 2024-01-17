# Stock-market-analysis
The project is a desktop stock market prediction app using Python, Eel, and yfinance. Users input a stock name or ticker, and the app fetches historical data to make predictions using deep learning. Results are displayed through a matplotlib plot, aiding in basic stock price forecasting and analysis.

The provided code is for a simple desktop application built using Eel in Python, allowing users to predict stock prices based on historical data. Here's a brief description of the code:

Libraries Used:

eel: A Python library for creating Electron-like desktop applications with web technologies.
yfinance: A library for fetching historical stock data from Yahoo Finance.
pandas: A data manipulation library in Python.
scikit-learn: A machine learning library for Python.
matplotlib: A plotting library for creating static, animated, and interactive visualizations.
Backend (app.py):

The Eel app is initialized, and a function (predict_stock) is exposed to the frontend.
The predict_stock function takes a stock ticker symbol as input, fetches historical stock data using yfinance, performs linear regression prediction, and plots the results using matplotlib.
Frontend (index.html and style.css):

The HTML file contains a simple form with an input field for entering the stock ticker symbol (stockTicker).
User inputs are processed using JavaScript, and the predictStock function is triggered when the "Predict Stock" button is clicked.
The result is displayed below the button.
Functionality:

Users enter a stock ticker symbol (e.g., AAPL for Apple) in the input field.
The application fetches historical stock data for the provided ticker using yfinance.
It performs a basic linear regression prediction on the closing stock prices.
The actual and predicted stock prices are plotted using matplotlib, and the plot is displayed to the user.
The application provides feedback to the user, indicating the success of the prediction or any errors encountered.
