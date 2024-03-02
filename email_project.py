import streamlit as st 


def project2():
    st.header("""Predicting Email Campaign Effectiveness using Deep Learning and Random Forest""")
    st.markdown("In today's digital age, email campaigns have become a vital tool for businesses to reach their target audience. Analyzing the effectiveness of these campaigns is crucial for optimizing marketing strategies and maximizing returns on investment. In this blog post, we'll explore how machine learning techniques, including deep learning and random forest, can be used to predict the effectiveness of email campaigns.")
    
    st.subheader("Data Collection and Preprocessing")
    st.markdown("We obtained a dataset containing information about various aspects of email campaigns, such as subject hotness score, customer location, and email status (e.g., opened, clicked, bounced). After loading the dataset, we performed data cleaning to remove any missing values.")
    
    st.code("""# Data Loading and Preprocessing
    data=pd.read_csv('/content/data_email_campaign.csv')
    data = data.dropna()
    """)
    
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Before diving into modeling, we conducted exploratory data analysis to gain insights into the distribution of key features such as customer location and subject hotness score.")
    
    st.code("""# EDA: Customer Location
    data.groupby('Customer_Location').size().plot(kind='barh')
    plt.gca().spines[['top', 'right',]].set_visible(False)

    # EDA: Subject Hotness Score
    data['Subject_Hotness_Score'].plot(kind='hist', bins=20, title='Subject Hotness Score')
    plt.gca().spines[['top', 'right',]].set_visible(False)
    """)
    
    st.header("Model Building")
    st.markdown("Feedforward Neural Network (Deep Learning)We constructed a feedforward neural network (FFNN) using TensorFlow's Keras API. The model consists of multiple dense layers with different activation functions, followed by an output layer with a sigmoid activation function for binary classification.")
    
    st.code("""# Build and Compile FFNN Model
    FF_model = Sequential()
    # Add Dense layers
    # Compile the model
    FF_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    """)
    
    st.header("Random Forest Classifier")
    st.markdown("In addition to deep learning, we employed a traditional machine learning algorithm, the Random Forest Classifier, to compare its performance with the FFNN model.")
    
    st.code("""# Build and Train Random Forest Classifier
    RF = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier())
    ])
    RF.fit(X_train, y_train)
    """)
    
    st.header("Model Evaluation")
    st.markdown("We split the dataset into training and testing sets and evaluated the performance of both models using accuracy as the metric.")
    
    st.code("""# Evaluate FFNN Model
    accuracy_FF = accuracy_score(y_test, y_pred_DL)

    # Evaluate Random Forest Classifier
    accuracy_RF = accuracy_score(y_test, y_pred_RF)
    """)
    
    st.header("Conclusion")
    st.markdown("In this blog post, we demonstrated how deep learning and traditional machine learning techniques can be applied to predict the effectiveness of email campaigns. Both the feedforward neural network and random forest classifier achieved impressive accuracy scores, highlighting their potential for optimizing marketing strategies and enhancing campaign performance.")
