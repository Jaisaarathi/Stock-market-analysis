import eel
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set up the Eel app
eel.init('main')

@eel.expose
def predict_stock(ticker,start,end):
    try:
        # Fetch historical stock data using yfinance
        stock_data = yf.download(ticker, start, end)

        # Extract features (date as days since the first date)
        ma_100_dayas=stock_data.Close.rolling(100).mean()
        ma_200_dayas=stock_data.Close.rolling(200).mean()

        # Split data into training and testing sets
        data_train=pd.DataFrame(stock_data.Close[0:int(len(stock_data)*0.80)])
        data_test=pd.DataFrame(stock_data.Close[int(len(stock_data)*0.80):len(stock_data)])
        
        from sklearn.preprocessing import MinMaxScaler
        scaler =MinMaxScaler(feature_range=(0,1))
        
        data_train_scale= scaler.fit_transform(data_train)

        # Split data into features (X) and target (y)
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

        # Predict the stock prices for the entire dataset
        y_predict = model.predict(x)
        scale = 1/scaler.scale_
        y_predict= y_predict*scale
        y=y*scale

        # Plot the predictions
        plt.plot(y_predict,'r',label='Predicted price')
        plt.plot(y,'g',label='Original price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        

        return 'Prediction successful. Check the plot.'
    except Exception as e:
        return f'Error: {str(e)}'

# Start the Eel app
eel.start('template\index.html', size=(800, 600))