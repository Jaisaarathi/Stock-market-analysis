import eel
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Set up the Eel app
eel.init('eel')

@eel.expose
def predict_stock(ticker,start,end):
    try:
        # Fetch historical stock data using yfinance
        stock_data = yf.download(ticker, start, end)
        
        ma_100_dayas=stock_data.Close.rolling(100).mean()
        ma_200_dayas=stock_data.Close.rolling(200).mean()
        
        data_train=pd.DataFrame(stock_data.Close[0:int(len(stock_data)*0.80)])
        data_test=pd.DataFrame(stock_data.Close[int(len(stock_data)*0.80):len(stock_data)])
        
        from sklearn.preprocessing import MinMaxScaler
        scaler =MinMaxScaler(feature_range=(0,1))
        
        data_train_scale= scaler.fit_transform(data_train)
        
        x=[]
        y=[]

        for i in range(100,data_train_scale.shape[0]):
            x.append(data_train_scale[i-100:i])
            y.append(data_train_scale[i,0])
        
        x,y=np.array(x),np.array(y)
        
        from keras.layers import Dense , Dropout, LSTM
        from keras.models import Sequential

        model=Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=((x.shape[1],1))))
        model.add(Dropout(0.2))
        model.add(LSTM(units=60, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.4))
        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=1))
        
        model.compile(optimizer = 'adam', loss='mean_squared_error')
        
        model.fit(x,y, epochs= 5, batch_size=32,verbose=1)
        
        pas_100_days = data_train.tail(100)
        data_test = pd.concat([pas_100_days, data_train],ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)
        
        x=[]
        y=[]

        for i in range(100,data_test_scale.shape[0]):
            x.append(data_test_scale[i-100:i])
            y.append(data_test_scale[i,0])
    
        x,y=np.array(x),np.array(y)
        
        y_predict = model.predict(x)
        
        scale = 1/scaler.scale_
        
        y_predict= y_predict*scale
        
        y=y*scale
        
        plt.plot(y_predict,'r',label='Predicted price')
        plt.plot(y,'g',label='Original price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        # Extract features (date as days since the first date)
        #stock_data.reset_index(inplace=True)
        #stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days

        # Split data into features (X) and target (y)
        #X = stock_data[['Days']].values
        #y = stock_data['Close'].values

        # Split data into training and testing sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a linear regression model
        #model = LinearRegression()
        #model.fit(X_train, y_train)

        # Predict the stock prices for the entire dataset
        #stock_data['Predictions'] = model.predict(X)

        # Plot the predictions
        #plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Close Price')
        #plt.plot(stock_data['Date'], stock_data['Predictions'], label='Predicted Close Price', linestyle='--')
        #plt.xlabel('Date')
        #plt.ylabel('Close Price')
        #plt.title(f'Stock Price Prediction for {ticker}')
        #plt.legend()
        #plt.show()

        return 'Prediction successful. Check the plot.'
    except Exception as e:
        return f'Error: {str(e)}'

# Start the Eel app
eel.start('gui\index.html', size=(800, 600))
