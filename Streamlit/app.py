import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Mental Health Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('Streamlit/clean_data.csv')

df = load_data()

# Sidebar navigation
page = st.sidebar.selectbox("Go to", ["About", "Predict Age", "Treatment Seeking", "Clustering"])

# About page
if page == 'About':
    st.header('Mental Health in Tech Industry Survey')
    st.write("This dashboard analyzes data from a 2014 survey about mental health in the tech workplace.")
    
    st.write("Dataset Source: [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)")
    
    st.subheader('Key Insights')
    st.write("- About 50% of respondents have sought treatment for mental health")
    st.write("- 35% report mental health sometimes interferes with work")
    
    st.subheader('Dataset Preview')
    st.dataframe(df.head())

# Predict Age page
elif page == 'Predict Age':
    st.header("Age Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        family_history = st.selectbox("Family history of mental illness?", ['Yes', 'No'])
        treatment = st.selectbox('Sought treatment for mental health?', ['Yes', 'No'])
    
    with col2:
        st.subheader("Work Environment")
        work_interfere = st.selectbox('Does mental health interfere with work?',
                                    ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
        remote_work = st.selectbox('Work remotely >50% time?', ['Yes', 'No'])
        benefits = st.selectbox('Mental health benefits?', ["Don't know", 'Yes', 'No'])
    
    if st.button('Predict Age'):
        try:
            model = joblib.load('reg_model.pkl')
            input_df = pd.DataFrame([{
                'Gender': gender,
                'family_history': family_history,
                'treatment': treatment,
                'work_interfere': work_interfere,
                'remote_work': remote_work,
                'benefits': benefits
            }])
            
            predicted_age = model.predict(input_df)
            predicted_age = np.expm1(predicted_age)[0]
            
            st.subheader(f"Predicted Age: {predicted_age:.1f} years")
            
            fig, ax = plt.subplots()
            sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=ax)
            ax.axvline(predicted_age, color='red', linestyle='--', label='Predicted Age')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Treatment Seeking page
elif page == "Treatment Seeking":
    st.header("Mental Health Treatment Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Factors")
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        family_history = st.selectbox("Family history of mental illness?", ['Yes', 'No'])
        work_interfere = st.selectbox('Does mental health interfere with work?',
                                    ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
    
    with col2:
        st.subheader("Workplace Factors")
        benefits = st.selectbox('Mental health benefits?', ["Don't know", 'Yes', 'No'])
        wellness_program = st.selectbox('Employer discusses mental health?', ["Don't know", 'Yes', 'No'])
    
    if st.button('Predict Treatment Seeking'):
        try:
            clf = joblib.load('clf_model.pkl')
            input_df = pd.DataFrame([{
                'Gender': gender,
                'family_history': family_history,
                'work_interfere': work_interfere,
                'benefits': benefits,
                'wellness_program': wellness_program
            }])
            
            prediction = clf.predict(input_df)[0]
            proba = clf.predict_proba(input_df)[0][1]
            
            if prediction == 1:
                st.success(f"Likely to seek treatment (confidence: {proba*100:.1f}%)")
            else:
                st.error(f"Unlikely to seek treatment (confidence: {(1-proba)*100:.1f}%)")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Clustering page
elif page == 'Clustering':
    st.header("Employee Clustering Analysis")
    
    st.subheader("Cluster Characteristics")
    
    st.write("""
    **Cluster 0: Supervisor-Reliant Onsite Workers**
    - Family history: Very low (2%)
    - Treatment: High (90%)
    - Work interference: Moderate (47%)
    """)
    
    st.write("""
    **Cluster 1: Treated but Employer-Wary**
    - Family history: Very low (2%)
    - Treatment: Very high (93%)
    - Work interference: High (60%)
    """)
    
    st.write("""
    **Cluster 2: Remote High-Risk Communicators**
    - Family history: Very low (1%)
    - Treatment: Moderate (74%)
    - Work interference: Moderate-High (52%)
    """)
