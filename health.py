import streamlit as st 
def health():
                st.header("""Predicting Health Insurance Cross-Sell Response with Machine Learning""")
                st.markdown("In the insurance industry, predicting customer behavior is essential for targeting potential clients and optimizing marketing strategies. In this blog post, we'll explore how machine learning techniques can be used to predict the response of customers to health insurance cross-selling.")
                st.subheader("Data Exploration and Preprocessing")
                st.markdown("We began by loading the dataset containing various features related to customers and their response to health insurance cross-selling. After splitting the data into features (X) and target variable (y), we further divided it into training and testing sets.")
                st.code("""# Data Loading and Preprocessing
                        df=pd.read_csv('/content/TRAIN-HEALTH INSURANCE CROSS SELL PREDICTION.csv')
                        X=df.drop(columns='Response')
                        y=df['Response']

                        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                        """)
                
                st.header("Feature Engineering and Selection")
                st.markdown("We employed feature engineering techniques to transform and select relevant features for training our model. This involved scaling numerical features and one-hot encoding categorical features.")
                st.code("""# Feature Engineering and Selection
                        num_features=['Annual_Premium','Vintage']
                        numeric_transformer= Pipeline(steps=[
                            ('scaler',StandardScaler())
                        ])

                        categorical_features=['Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
                        categorical_transformer=Pipeline(steps=[
                            ('onehot',OneHotEncoder(handle_unknown='ignore'))
                        ])

                        preprocessor=ColumnTransformer(
                            transformers=[
                                ('num',numeric_transformer,num_features),
                                ('cat',categorical_transformer,categorical_features)
                            ])

                             """)
                
                st.header("Model Building and Evaluation")
                st.markdown("We constructed a machine learning pipeline consisting of preprocessing steps, feature selection using SelectKBest, and a logistic regression classifier. The model was trained on the training data and evaluated based on accuracy score using the test data.")
                st.code("""# Model Building and Evaluation
                        pipeline = Pipeline(steps=[
                            ('preprocessor',preprocessor),
                            ('feature_selection',SelectKBest(score_func=f_classif,k=5)),
                            ('classifier',LogisticRegression())
                        ])

                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)

                        accuracy = accuracy_score(y_test, y_pred)
                        print("Accuracy:", accuracy)

                        """)
                
                st.header("Conclusion")
                st.markdown("In this blog post, we demonstrated how machine learning techniques can be leveraged to predict the response of customers to health insurance cross-selling. By preprocessing and selecting relevant features and building a logistic regression classifier, we achieved a commendable accuracy score, indicating the model's potential for aiding insurance companies in targeting potential clients effectively.")
                