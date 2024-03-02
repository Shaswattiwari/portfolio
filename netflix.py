import streamlit as st 
def netflix():
                st.header("""Understanding Content Trends on Netflix Through Unsupervised Machine Learning""")
                st.markdown("In recent years, Netflix has become a powerhouse in the entertainment industry, offering a vast library of movies and TV shows to subscribers worldwide. In this blog post, we delve into a fascinating project aimed at understanding the type of content available on Netflix in different countries and examining whether Netflix has been shifting its focus towards TV shows over movies. Additionally, we explore the clustering of similar content using text-based features.")
                st.subheader("Exploring Content Availability Across Countries")
                st.markdown("One of the key aspects of our analysis involves understanding the distribution of content types (movies vs. TV shows) across various countries. Leveraging data from a Netflix dataset, we embarked on visualizing this distribution through a choropleth map. The map provides an insightful overview of the number of movies and TV shows available in different countries, allowing us to identify regions where one type of content might be more prevalent than the other.")
                st.code("""# Handling missing values
                            data.loc[:, ['director', 'cast', 'country', 'date_added']] = data.loc[:, ['director', 'cast', 'country', 'date_added']].fillna('unknown')
                            # Data preprocessing for visualization
                            working_df = data
                            working_df_split = working_df.assign(country=working_df['country'].str.split(', ')).explode('country')
                            content_type_among_country = working_df_split.groupby(['country', 'type']).size().unstack(fill_value=0).reset_index()
                            # Visualization 1: Choropleth Map
                            fig = px.choropleth(content_type_among_country,
                                                locations='country',
                                                locationmode='country names',
                                                color='Movie',
                                                hover_name='country',
                                                hover_data={'Movie': True, 'TV Show': True},
                                                color_continuous_scale='Darkmint',
                                                labels={'Movie': 'Number of Movies', 'TV Show': 'Number of TV Shows'},
                                                title='Number of Movies and TV Shows by Country',
                                                projection='natural earth')
                            fig.update_geos(showcountries=True, countrycolor="Black")
                            fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'),
                                            margin=dict(l=0, r=0, t=0, b=50),
                                            width=1500,
                                            height=600)
                            fig.show()
                            """)
                                                                    
                st.header("Analyzing Netflix's Content Focus Over Time")
                st.markdown("""Next, we set out to investigate whether Netflix has been increasingly focusing on TV shows compared to movies in recent years. By grouping the data by release year and content type, we created a time series plot showcasing the count of movies and TV shows released each year. This analysis offers valuable insights into Netflix's content strategy evolution over time, highlighting potential shifts in emphasis between movies and TV shows.""")
                st.code(""" working_dfcontent_type_by_year = working_df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
                                                    release_year = content_type_by_year.index
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=release_year, y=content_type_by_year['Movie'], mode='lines', name='Movies'))
                            fig.add_trace(go.Scatter(x=release_year, y=content_type_by_year['TV Show'], mode='lines', name='TV Shows'))

                            fig.update_layout(title='Count of Movies and TV Shows by Release Year', xaxis_title='Release Year', yaxis_title='Count')
                            fig.show()
                        
                        """)
                
                st.header("Clustering Similar Content for Enhanced Recommendations")
                st.markdown("""In the era of personalized recommendations, clustering similar content can significantly enhance user experience on streaming platforms like Netflix. To achieve this, we employed unsupervised machine learning techniques, specifically K-means clustering, on textual features extracted from movie and TV show descriptions. By preprocessing the text and vectorizing it using TF-IDF, we obtained clusters of similar content, enabling Netflix to offer more targeted recommendations tailored to individual user preferences.""")
                st.code("""# Data preprocessing for clustering
                            to_be_cluster = working_df[['show_id', 'description']]
                            nltk.download('stopwords')
                            nltk.download('wordnet')
                            ENGLISH_STOP_WORDS = set(stopwords.words('english'))
                            lemmatizer = WordNetLemmatizer()


                            # Text preprocessing function
                            def preprocess_text(text):
                                text = text.lower()
                                text = re.sub(r'[^\w\s]', '', text)
                                text = re.sub(r'\d+', 'NUMBER', text)
                                text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
                                text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
                                return text
                                
                                # Applying text preprocessing
                            descriptions = to_be_cluster.description.apply(preprocess_text)
                            # Vectorizing text data
                            vectorizer = TfidfVectorizer()
                            X = vectorizer.fit_transform(descriptions)
                            # Finding optimal number of clusters
                            optimalK = OptimalK(parallel_backend='joblib')
                            n_clusters = optimalK(X.toarray(), cluster_array=np.arange(1, 11))
                            print("Optimal number of clusters:", n_clusters)
                            # Clustering using KMeans
                            kmeans = KMeans(n_clusters=n_clusters, n_init=5, max_iter=500, random_state=42)
                            kmeans.fit(X)
                            to_be_cluster['cluster'] = kmeans.labels_
                            clustered_data = to_be_cluster.drop(columns='description')
                            """)                        
                                        
                st.header("Conclusion")
                st.markdown("""Through this project, we gained valuable insights into the distribution of Netflix content across countries, analyzed trends in content focus over time, and implemented clustering techniques for enhancing content recommendations. By leveraging data-driven approaches, Netflix can continue to refine its content strategy and deliver a more personalized and engaging experience to its global audience.
                            In conclusion, the intersection of data science and entertainment opens up exciting possibilities for understanding audience preferences and shaping content delivery in the digital age.""")