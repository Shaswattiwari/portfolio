import streamlit as st 
def bike_sharing():
                st.header("""Predicting Bike Rental Demand in Seoul""")
                st.markdown("In this project, we aim to predict bike rental demand in Seoul, South Korea, using machine learning techniques. We will explore various models, including GRU, CNN, and LSTM, to forecast future bike rental counts based on weather and seasonal factors")
                st.subheader("Data Preprocessing")
                st.markdown("We started by loading the dataset and examining its structure. The dataset includes information such as the number of rented bikes, temperature, humidity, wind speed, and more. After loading the dataset, we performed the following preprocessing steps:Converted the 'Date' column to datetime format.Aggregated the data on a daily basis, considering average values for weather-related features and total rented bike count.Rounded numerical values to two decimal places for consistency.")
                st.code("""# Convert 'Date' column to datetime format
                        data['Date'] = pd.to_datetime(data['Date'],format='%d/%m/%Y')
                        # Aggregate data on a daily basis
                        data = data.groupby('Date').agg({
                            'Rented Bike Count': 'sum',
                            'Hour': 'mean',  # You might want to use another aggregation method for 'Hour'
                            'Temperature(°C)': 'mean',
                            'Humidity(%)': 'mean',
                            'Wind speed (m/s)': 'mean',
                            'Visibility (10m)': 'mean',
                            'Dew point temperature(°C)': 'mean',
                            'Solar Radiation (MJ/m2)': 'mean',
                            'Rainfall(mm)': 'sum',
                            'Snowfall (cm)': 'sum',
                            'Seasons': 'first',  # Assuming seasons don't change within a day
                            'Holiday': 'first'   # Assuming holiday doesn't change within a day
                        }).reset_index()
                        # Round numerical values to two decimal places
                        data=data.round(2)
                        # Explore the time series data
                        train_dates = pd.to_datetime((data['Date']))
                        """)
                
                st.header("Model Building")
                st.markdown("""We experiment with three different models to forecast bike rental demand:Gated Recurrent Unit (GRU) Model: A type of recurrent neural network (RNN) that is well-suited for sequence prediction tasks.Convolutional Neural Network (CNN) Model: Often used for analyzing visual imagery but can also be applied to time-series data.Long Short-Term Memory (LSTM) Model: A type of RNN that can capture long-term dependencies in time-series data.Each model is trained using the scaled input-output pairs generated from the preprocessed data. We utilize Mean Squared Error (MSE) as the loss function and Adam optimizer for training the models.""")
                st.code("""from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import GRU, LSTM, Conv1D, Dense, Dropout
                        from tensorflow.keras.optimizers import Adam

                        # Define and compile GRU model
                        gru = Sequential([
                            GRU(64, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                            Dropout(0.2),
                            GRU(32, activation='relu'),
                            Dropout(0.2),
                            Dense(y.shape[1])
                        ])
                        gru.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                        history_gru = gru.fit(X, y, epochs=5)

                        # Define and compile CNN model
                        cnn = Sequential([
                            Conv1D(64, activation='relu', kernel_size=3, input_shape=(X.shape[1], X.shape[2])),
                            Dropout(0.2),
                            Conv1D(32, kernel_size=3, activation='relu'),
                            Dropout(0.2),
                            Dense(y.shape[1])
                        ])
                        cnn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                        history_cnn = cnn.fit(X, y, epochs=5)

                        # Define and compile RNN model
                        rnn = Sequential([
                            LSTM(32, activation="relu", input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
                            LSTM(16, activation="relu"),
                            Dropout(0.1),
                            Dense(y.shape[1])
                        ])
                        rnn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                        history_rnn = rnn.fit(X, y, epochs=5)

                        """
                        )
                
                st.header("Forecasting")
                st.markdown("After training the models, we make predictions for future bike rental demand. We forecast the bike rental counts for the next 30 days using each model. The predicted values are then visualized alongside known data to evaluate the forecasting performance.")
                st.code("""# # Make predictions using each model
                        predictions_model1 = gru.predict(X[-day_in_future:])
                        predictions_model2 = cnn.predict(X[-day_in_future:])
                        predictions_model3 = rnn.predict(X[-day_in_future:])
                        predictions_model3=scaler.inverse_transform(predictions_model3)
                        predictions_model1=scaler.inverse_transform(predictions_model1)
                        predictions_model2=scaler.inverse_transform(np.squeeze(predictions_model2))
                        # Convert them into pandas DataFrame objects
                        df1 = pd.DataFrame(predictions_model1).iloc[:,:1].round(0)
                        df2 = pd.DataFrame(predictions_model2).iloc[:,:1].round(0)
                        df3 = pd.DataFrame(predictions_model3).iloc[:,:1].round(0)
                        combined_df = pd.concat([df1, df2, df3],axis=1)
                        predictions = combined_df.mean(axis=1).round(0)
                        """)
                                        
                st.header("Conclusion")
                st.markdown("In conclusion, this project demonstrates the application of machine learning techniques to predict bike rental demand based on weather and seasonal factors. By preprocessing the data, building and training machine learning models, and evaluating their performance, we gain insights into forecasting future bike rental demand in Seoul.")
                