import streamlit as st 
def yes():
                st.header("""Predicting Stock Prices: YES Bank""")
                st.markdown("In this project, we aim to predict the closing stock prices of YES Bank using machine learning techniques. We will explore data preprocessing, feature engineering, model building, and evaluation techniques to forecast future stock prices accurately.")
                st.subheader("Data Preprocessing")
                st.markdown("We start by loading the historical stock price data of YES Bank. The dataset contains information about the opening, high, low, closing prices, adjusted closing prices, and volume of the stock. We preprocess the data by converting the 'Date' column to datetime format and extracting additional features such as lagged prices and interaction features.")
                st.code("""train_data=stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
                        """)
                                        
                st.header("Feature Engineering")
                st.markdown("""We engineer features by creating lagged features for each of the main attributes (Open, High, Low, Close, Adj Close, Volume) with different lag periods (1, 7, and 14 days). Additionally, we create interaction features between the closing price and other attributes to capture potential relationships.""")
                st.code(""" # Create lag features
                        lag_periods = [1, 7, 14]
                        for lag_period in lag_periods:
                            for column in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                                train_data['{column}_{lag_period}day_lag'] = train_data[column].shift(lag_period)
                                train_data['Open_close_interaction'] = train_data['Close'] * train_data['Open']
                        train_data['high_close_interaction'] = train_data['Close'] * train_data['High']
                        train_data['low_close_interaction'] = train_data['Close'] * train_data['Low']
                        train_data['Adj_Close_interaction'] = train_data['Close'] * train_data['Adj Close']
                        train_data['Volume_close_interaction'] = train_data['Close'] * train_data['Volume']
                        train_data.dropna(inplace=True)
                        """)
                
                st.header("Data Scaling")
                st.markdown("""We scale the engineered features using MinMaxScaler to bring all feature values within the range [0, 1]. This normalization step is essential for training neural network models.""")
                st.code("""scaler=MinMaxScaler()
                        scaler=scaler.fit(train_data)
                        scaled_train_data=scaler.transform(train_data)""")                        
                                        
                st.header("Model Building")
                st.markdown("""We build a Long Short-Term Memory (LSTM) neural network model using the Sequential API from Keras. The model architecture consists of two LSTM layers with ReLU activation functions, followed by a dropout layer for regularization and a dense output layer.""")
                st.code("""optimizer = Adam(learning_rate=0.1)
                        model=Sequential()
                        model.add(LSTM(64,activation="relu",input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
                        model.add(LSTM(32,activation="relu",return_sequences=False))
                        model.add(Dropout(0.2))
                        model.add(Dense(y.shape[1]))

                        model.compile(optimizer=optimizer,loss='mean_squared_error')

                        """)
                                        
                st.header("Forecasting")
                st.markdown("""We make predictions for future stock prices using the trained model. We forecast the closing prices for the next 29 days based on the last available data point..""")
                
                st.header("Visualization and Analysis")
                st.markdown("""We visualize the predicted stock prices alongside the known historical data to analyze the model's performance. Additionally, we calculate and visualize the residuals (the differences between actual and predicted values) to assess the model's accuracy.""")
                
                st.header("Conclusion")
                st.markdown("In conclusion, this project demonstrates the application of LSTM neural networks for predicting stock prices. By preprocessing the data, engineering relevant features, building and training a neural network model, and evaluating its performance, we gain insights into forecasting future stock prices of YES Bank.")
                