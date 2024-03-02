import streamlit as st 
def airline():
                st.header("""Analyzing Airline Reviews: A Sentiment Analysis Approach""")
                st.markdown("In this blog post, we delve into the world of airline reviews, exploring how sentiment analysis and machine learning can provide insights into customer satisfaction and recommendations. We'll walk through the process of building a sentiment analysis model and using it to predict whether a given review is likely to recommend the airline or not.")
                st.subheader("Data Collection and Preprocessing")
                st.markdown("We collected airline review data from various sources and cleaned it to remove any missing values. The dataset includes features such as overall rating, seat comfort, cabin service, food and beverage quality, entertainment options, ground service, and value for money.")
                st.subheader("Sentiment Analysis")
                st.markdown("Using the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool, we performed sentiment analysis on the customer reviews to extract the overall sentiment expressed in each review. VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.")
                st.code("""# Sentiment Analysis Function
                            def preprocess_text(text):
                            # Convert to lowercase
                            text = text.lower()
                            # Remove punctuation
                            text = re.sub(r'[^\w\s]', '', text)
                            # Replace numbers with placeholder
                            text = re.sub(r'\d+', 'NUMBER', text)
                            # Remove stopwords
                            text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
                            # Lemmatization
                            lemmatizer = WordNetLemmatizer()
                            text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
                            # Calculate sentiment score
                            analyzer = SentimentIntensityAnalyzer()
                            score = analyzer.polarity_scores(text)['compound']
                            return score

                            data['review_sentiment'] = data['customer_review'].apply(preprocess_text)""")
                
                st.header("Model Training and Evaluation")
                st.markdown("We split the dataset into training and testing sets and trained a logistic regression model using features such as overall rating, seat comfort, cabin service, etc. The model achieved an accuracy of approximately 85% on the test data.")
                st.code("""# Model Training
                        X = data[['overall', 'seat_comfort', 'cabin_service', 'food_bev', 'entertainment', 'ground_service', 'value_for_money', 'review_sentiment']]
                        y = data['recommended']
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        model = LogisticRegression()
                        model = model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        print("Accuracy:", accuracy)
                        """)
                
                st.header("Prediction Function")
                st.markdown("We created a prediction function that takes a new review as input along with various rating parameters and predicts whether the review is likely to recommend the airline or not.")
                st.code("""# Prediction Function
                            def predict_recommendation(review_text, overall_rating, seat_comfort, cabin_service, food_bev, entertainment, ground_service, value_for_money):
                            processed_text = preprocess_text(review_text)
                            new_review_df = pd.DataFrame({
                                'overall': [overall_rating],
                                'seat_comfort': [seat_comfort],
                                'cabin_service': [cabin_service],
                                'food_bev': [food_bev],
                                'entertainment': [entertainment],
                                'ground_service': [ground_service],
                                'value_for_money': [value_for_money],
                                'review_sentiment': [processed_text]
                            })
                            recommendation = model.predict(new_review_df)[0]
                            if recommendation == 1:
                                return "likely to be recommended."
                            else:
                                return "not likely to be recommended."
                        """)
                
                st.header("Example Usage")
                st.markdown("Let's see how our prediction function works with an example review:")
                st.code("""new_review_text = "worst flight ever, the staff was not friendly"
                            predicted_recommendation = predict_recommendation(new_review_text, overall_rating=3, seat_comfort=2, cabin_service=0, food_bev=4, entertainment=2, ground_service=1, value_for_money=5)
                            print(predicted_recommendation)
                            """)
                
                st.markdown("In this example, the review is predicted to be not likely to be recommended.")
                st.markdown("By leveraging sentiment analysis and machine learning, we can gain valuable insights from customer reviews to improve airline services and customer satisfaction.")
                