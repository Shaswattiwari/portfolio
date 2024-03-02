import streamlit as st 
from airline import airline
from streamlit_option_menu import option_menu
from email_project import project2
from health import health
from bike import bike_sharing
from tata import tata
from yes_bank import yes
from netflix import netflix
from tele_gram1 import gram

def projects():

    with st.container():
        with st.expander('Classification -'):       

            selected_project1 = option_menu(
                menu_title=None,
                options=["Airline Passenger Referral Prediction", 
                         "Email Campaign Effectiveness Prediction",
                         "Health Insurance Cross Sell Prediction"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal"
            )

            if selected_project1 == "Airline Passenger Referral Prediction":
                airline()
            elif selected_project1 == "Email Campaign Effectiveness Prediction":
                project2()
            elif selected_project1 == "Health Insurance Cross Sell Prediction":
                health()

    with st.container():
        with st.expander('Regression -'):
            
            selected_project2 = option_menu(
                menu_title=None,
                options=["Bike Sharing Demand Prediction",
                         "Tata steel Stock Closing Price forecasting",
                         "Yes Bank Stock Closing Price forecasting"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal"
            )

            if selected_project2 == "Bike Sharing Demand Prediction":
                bike_sharing()
            elif selected_project2 == "Tata steel Stock Closing Price forecasting":
                tata()
            elif selected_project2 == "Yes Bank Stock Closing Price forecasting":
                yes()                        

    with st.container():
        with st.expander('Unsupervised ML - Netflix Movies and TV Shows Clustering'):
            netflix()
            
    with st.container():
        with st.expander('telebot'):
            gram()
            
            
    with st.container():
        with st.expander('SQL Query Generator & Executor with LLM (In Development)'):
            st.subheader("Description:")
            st.markdown("ongoing project is the development of a software tool utilizing a Language Model (LLM) trained on SQL to swiftly generate and execute SQL queries via SQLite 3.")
            st.subheader("Key features include:")
            st.markdown("""
                        LLM Integration: Interpret natural language input to generate accurate SQL queries.
                        
                        Query Optimization: Enhance query efficiency for improved performance.
                        
                        SQL Execution:Execute queries seamlessly using SQLite 3.User-Friendly Interface: Intuitive controls for easy interaction.
                        
                        Query History: Maintain history for tracking and versioning.
                        
                        Customization: Tailor preferences for specific needs.
                        """)
            st.markdown("This project aims to revolutionize database interaction, empowering users with efficient SQL query management for various applications. Currently, we are in the development phase, working on implementing and refining these features to deliver a robust and user-friendly solution.")
            
            
    with st.container():
        with st.expander('SResume Builder with Streamlit (In Development)'):
            st.subheader("Key features include:")
            st.markdown("""
                        Interactive Interface: Utilizing Streamlit's capabilities, the application will offer an interactive and intuitive interface for users to input their resume details.
                        
                        Customization Options: Users will have the flexibility to customize various elements of their resumes, including layout, font styles, and content sections.
                        
                        Template Selection: The application will provide a selection of pre-designed templates for users to choose from, catering to different industries and job roles.
                        
                        Dynamic Preview: Streamlit's live preview feature will allow users to see real-time updates to their resumes as they make changes, ensuring a seamless editing experience.
                        
                        Export Options: Once the resume is finalized, users will have the option to export it in multiple formats, including PDF and Word, for easy sharing and printing.
                        
                        """)
            st.markdown("As the project is currently in development, I am actively working on implementing these features and refining the application to deliver a high-quality and user-friendly Resume Builder with Streamlit.")
            
    with st.container():
        with st.expander('Power BI Dashboard  (In Development)'):
            st.subheader("Key features include:")
            st.markdown("""
                        Data Integration: Integration of diverse data sources into Power BI for comprehensive data analysis.
                        
                        Visualization Design: Designing visually appealing and informative charts, graphs, and visualizations to represent data effectively.
                        
                        Interactive Dashboard: Creating interactive dashboards with drill-down capabilities, allowing users to explore data at different levels of detail.

                        Data Modeling: Implementing data modeling techniques within Power BI to optimize data relationships and calculations.
                        
                        Real-time Updates: Incorporating real-time data streaming and automatic refresh functionalities to ensure that the dashboard reflects the most up-to-date information.
                        
                        Collaboration and Sharing: Enabling collaboration among team members by sharing and publishing dashboards securely within the organization.
                        """)
            st.markdown("As the project is currently in development, I am actively working on implementing these features and fine-tuning the Power BI Dashboard to meet the specific needs and requirements of the end users.")
            