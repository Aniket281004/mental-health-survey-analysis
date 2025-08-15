import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Mental Health Analysis", layout="wide")
age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
# Load data
@st.cache_data
def load_data():
    return pd.read_csv('Streamlit/clean_data.csv')

df = load_data()

# Sidebar navigation
page = st.sidebar.selectbox("Go to", ["About", "Predict Age", "Treatment Prediction", "Clustering"])

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
    
    # Input features
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox('Have you sought treatment for a mental health condition?', ['Yes', 'No'])
    work_interfere = st.selectbox('If you have a mental health condition, do you feel that it interferes with your work?',
                                ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
    
    st.subheader("Work Environment")
    remote_work = st.selectbox('Do you work remotely (outside of an office) at least 50% of the time?', ['Yes', 'No'])
    benefits = st.selectbox('Does your employer provide mental health benefits?', ["Don't know", 'Yes', 'No'])
    care_options = st.selectbox('Do you know the options for mental health care your employer provides?',
                              ['Not sure', 'No', 'Yes'])
    wellness_program = st.selectbox('Has your employer ever discussed mental health as part of a wellness program?',
                                  ["Don't know", 'Yes', 'No'])
    seek_help = st.selectbox('Does your employer provide resources to learn about mental health and seeking help?',
                           ['Yes', 'No'])
    leave = st.selectbox('How easy is it for you to take mental health leave?',
                       ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
    
    st.subheader("Attitudes and Perceptions")
    mental_health_consequence = st.selectbox('Would discussing mental health with your employer have negative consequences?',
                                           ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox('Would you discuss a mental health issue with your coworkers?',
                           ['Some of them', 'No', 'Yes'])
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?',
                                         ['No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                            ['No', 'Maybe', 'Yes'])

    if st.button('Predict Age'):
        try:
            model = joblib.load('Streamlit/reg_model.pkl')
            input_df = pd.DataFrame([{
               
                'Gender': gender,
                'self_employed': self_employed,
                'family_history': family_history,
                'treatment': treatment,
                'work_interfere': work_interfere,
                'remote_work': remote_work,
                'benefits': benefits,
                'care_options': care_options,
                'wellness_program': wellness_program,
                'seek_help': seek_help,
                'leave': leave,
                'mental_health_consequence': mental_health_consequence,
                'coworkers': coworkers,
                'mental_health_interview': mental_health_interview,
                'supervisor': supervisor
            }])
            
            predicted_age = model.predict(input_df)
            predicted_age = np.expm1(predicted_age)[0]
            
         
    
    # Round and clip to valid age group index
            pred_rounded = int(np.round(predicted_age))
            pred_rounded = np.clip(pred_rounded, 0, len(age_labels)-1)
    
    # Map to label
            predicted_age = age_labels[pred_rounded]
    
    # Display
            st.subheader("Result")
            st.write(f"Predicted Age Group: **{predicted_age}**")
    
    # Debug info (optional)
            st.caption(f"Raw regression output: {pred_encoded:.2f} → Rounded to index: {pred_rounded}")
            
            fig, ax = plt.subplots()
            sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=ax)
            ax.axvline(predicted_age, color='red', linestyle='--', label='Predicted Age')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Treatment Prediction page
elif page == "Treatment Prediction":
    st.header("Mental Health Treatment Prediction")
    
    # Input features
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    work_interfere = st.selectbox('If you have a mental health condition, do you feel that it interferes with your work?',
                                ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
    
    st.subheader("Work Environment")
    remote_work = st.selectbox('Do you work remotely (outside of an office) at least 50% of the time?', ['Yes', 'No'])
    benefits = st.selectbox('Does your employer provide mental health benefits?', ["Don't know", 'Yes', 'No'])
    care_options = st.selectbox('Do you know the options for mental health care your employer provides?',
                              ['Not sure', 'No', 'Yes'])
    wellness_program = st.selectbox('Has your employer ever discussed mental health as part of a wellness program?',
                                  ["Don't know", 'Yes', 'No'])
    seek_help = st.selectbox('Does your employer provide resources to learn about mental health and seeking help?',
                           ['Yes', 'No'])
    leave = st.selectbox('How easy is it for you to take mental health leave?',
                       ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
    
    st.subheader("Attitudes and Perceptions")
    mental_health_consequence = st.selectbox('Would discussing mental health with your employer have negative consequences?',
                                           ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox('Would you discuss a mental health issue with your coworkers?',
                           ['Some of them', 'No', 'Yes'])
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?',
                                         ['No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                            ['No', 'Maybe', 'Yes'])

    if st.button('Predict Treatment Seeking'):
        try:
            clf = joblib.load('Streamlit/clf_model.pkl')
            input_df = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'self_employed': self_employed,
                'family_history': family_history,
                'work_interfere': work_interfere,
                'remote_work': remote_work,
                'benefits': benefits,
                'care_options': care_options,
                'wellness_program': wellness_program,
                'seek_help': seek_help,
                'leave': leave,
                'mental_health_consequence': mental_health_consequence,
                'coworkers': coworkers,
                'mental_health_interview': mental_health_interview,
                'supervisor': supervisor
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
