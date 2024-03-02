import streamlit as st 
def tata():
                st.header("""Predicting Stock Prices: Tata Motors""")
                st.markdown("In this project, we aim to predict the closing stock prices of Tata Motors using machine learning techniques. We will explore feature selection, data preprocessing, model building, and evaluation techniques to forecast the future stock prices.")
                st.subheader("Data Preprocessing")
                st.markdown("We start by loading the dataset, which contains historical stock price data of Tata Motors. We preprocess the data by:Converting the 'Date' column to datetime format.Extracting additional features such as 'Year', 'Month', and 'Day' from the date.Dropping irrelevant columns such as 'Adj Close' and setting the 'Date' column as the index.")
                st.code("""tata_moters['Date']=pd.to_datetime(tata_moters['Date'])
                        tata_moters['Year'] = tata_moters['Date'].apply(lambda x: pendulum.instance(x).year)
                        tata_moters['Month'] = tata_moters['Date'].apply(lambda x: pendulum.instance(x).month)
                        tata_moters['Day'] = tata_moters['Date'].apply(lambda x: pendulum.instance(x).day)
                        numeric_columns = tata_moters.select_dtypes(include=[np.number])

                        # Initialize the MinMaxScaler
                        scaler = MinMaxScaler()

                        # Fit and transform the numeric columns
                        scaled_data = scaler.fit_transform(numeric_columns)

                        # Convert the scaled data back to a DataFrame
                        numeric_columns = pd.DataFrame(scaled_data, columns=numeric_columns.columns)

                        # Display the scaled data
                        numeric_columns = tata_moters.select_dtypes(include=[np.number])
                        X = numeric_columns.drop(columns=['Close'])  # Features
                        y = numeric_columns['Close']  # Target variable
                        """)
                                        
                st.header("Feature Selection")
                st.markdown("""We perform feature selection using the f_regression method to select the top k features that are most relevant to predicting the closing stock prices. The selected features are used for model training.""")
                st.code(""" Select top k features based on f_regression
                        k = 6  # Number of top features to select
                        selector = SelectKBest(score_func=f_regression, k=k)
                        X_selected = selector.fit_transform(X, y)

                        # Get selected feature indices
                        selected_feature_indices = selector.get_support(indices=True)

                        # Get selected feature names
                        selected_feature_names = X.columns[selected_feature_indices]

                        # Display selected feature names
                        print("Selected feature names:")
                        print(selected_feature_names)
                        """)
                
                st.header("Model Building")
                st.markdown("""We build a neural network model using the Sequential API from Keras. The model consists of three densely connected layers with ReLU activation functions. We compile the model using the Adam optimizer and mean squared error as the loss function.""")
                st.code("""model=Sequential([Dense(64,activation='relu',input_shape=(6,)),
                  Dense(32,activation='relu'),
                  Dense(1)])

                model.compile(optimizer='adam',loss='mean_squared_error')""")
                                        
                st.header("Visualization and Analysis")
                st.markdown("""We visualize the predicted stock prices alongside the actual stock prices to analyze the model's performance. Additionally, we analyze the residuals (the differences between actual and predicted values) using various plots such as residual plot, autocorrelation function (ACF) plot, partial autocorrelation function (PACF) plot, histogram of residuals, and Q-Q plot.""")
                
                
                st.header("Conclusion")
                st.markdown("In conclusion, this project demonstrates the application of machine learning techniques to predict stock prices. By preprocessing the data, selecting relevant features, building and training a neural network model, and evaluating its performance, we gain insights into forecasting future stock prices of Tata Motors.")
                