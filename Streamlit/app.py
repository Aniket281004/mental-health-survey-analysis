import streamlit as st
import numpy as np
import joblib 
import pandas as pd



page = st.sidebar.selectbox("Go to", ["About","Predict Age","Treatment Seeking Employees","Clustering Report"])

if page == 'About':
    st.header('Dataset Overview')
    st.markdown('[Dataset Used : ] , https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey')
    st.write('This dataset is from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders in the tech workplace.')
    
    st.header('Skewness in Age !')
  
    st.write('The dataset is not in particular an ideal option for regression due to its skew score of 1.05.')
    st.image(r'Images/Skewness.png')
    




elif page == 'Predict Age':
    st.header("📊 Age Prediction")
    st.subheader("Predicting age of employee using a Linear Regression model, based on their responses to certain questions about their mental health.")
    st.caption('(Acknowledgement: This dataset is not ideal for predicting age using regression. This is for research purposes only.)')

    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox('Have you sought treatment for a mental health condition?', ['Yes', 'No'])
    work_interfere = st.selectbox('If you have a mental health condition, do you feel that it interferes with your work?',
                                  ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
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
    mental_health_consequence = st.selectbox('Would discussing mental health with your employer have negative consequences?',
                                             ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox('Would you discuss a mental health issue with your coworkers?',
                             ['Some of them', 'No', 'Yes'])
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?',
                                           [ 'No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                              ['No', 'Maybe', 'Yes'])
    
   # Feature order must match training
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
   
    
# This will work with the pipeline
    if st.button('Predict'):
        model = joblib.load('Streamlit/reg_model.pkl')
        predicted_age = model.predict(input_df)
        st.write(f"Predicted Age: {np.expm1(predicted_age)} years")

if page == "Treatment Seeking Employees":
    st.header("Treatment Prediction")
    st.subheader('Predicting whether a employee is likely to seek mental health treatment')
    st.caption("Model Used : RandomForestClassifier")
    
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox('Have you sought treatment for a mental health condition?', ['Yes', 'No'])
    work_interfere = st.selectbox('If you have a mental health condition, do you feel that it interferes with your work?',
                                  ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown'])
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
    mental_health_consequence = st.selectbox('Would discussing mental health with your employer have negative consequences?',
                                             ['No', 'Maybe', 'Yes'])
    coworkers = st.selectbox('Would you discuss a mental health issue with your coworkers?',
                             ['Some of them', 'No', 'Yes'])
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?',
                                           [ 'No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                              ['No', 'Maybe', 'Yes'])
    
   # Feature order must match training
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

    
    if st.button('Predict'):
        clf = joblib.load('Streamlit/clf_model.pkl')
        predicted_treatment = clf.predict(input_df)
        if predicted_treatment == 1 :
            st.write('Yes')
        else :
            st.write("No")


if page == 'Clustering Report':
    st.header("Clustering employees")
    st.subheader("Clustering employees based on their nature towards seeking mental health.")
    st.write('I have used an k means algorithm with dimensionality reduction to make these clusters. ')
    st.image(r'Images/Clusters.png')
    st.header('Cluster Interpretation')
    st.subheader('Cluster 0: Supervisor-Reliant Onsite Workers')
    st.markdown('''
- Family history: Very low (2%)  
- Treatment: High (90%)  
- Work interference: Moderate (47%)  
- Remote work: None  
- Communication: Open to supervisors (58%), moderate to coworkers/employers  
''')

    st.subheader('Cluster 1: Treated but Employer-Wary')
    st.markdown('''
- Family history: Very low (2%)  
- Treatment: Very high (93%)  
- Work interference: High (60%)  
- Remote work: None  
- Communication: Moderate with coworkers and supervisors, low with employers  
''')

    st.subheader('Cluster 2: Remote High-Risk Communicators')
    st.markdown('''
- Family history: Very low (1%)  
- Treatment: Moderate (74%)  
- Work interference: Moderate-High (52%)  
- Remote work: Fully remote  
- Communication: Open across all levels  
''')



