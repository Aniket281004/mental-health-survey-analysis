import streamlit as st
import pandas as pd
import pickle

# ------------------------
# Load Models
# ------------------------
@st.cache_resource
def load_models():
    with open("age_regressor.pkl", "rb") as f:
        age_model = pickle.load(f)
    with open("treatment_classifier.pkl", "rb") as f:
        treatment_model = pickle.load(f)
    return age_model, treatment_model

age_model, treatment_model = load_models()

# ------------------------
# Sidebar Navigation
# ------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["About", "Age Prediction", "Treatment Prediction", "Clustering Report"]
)

# ------------------------
# Input for Age Prediction
# ------------------------
def inputs_for_age_prediction():
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    treatment = st.selectbox('Have you sought treatment for a mental health condition?', ['Yes', 'No'])
    work_interfere = st.selectbox(
        'If you have a mental health condition, do you feel that it interferes with your work?',
        ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown']
    )
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
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?', ['No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                              ['No', 'Maybe', 'Yes'])

    return pd.DataFrame({
        'Gender': [gender],
        'self_employed': [self_employed],
        'family_history': [family_history],
        'treatment': [treatment],
        'work_interfere': [work_interfere],
        'remote_work': [remote_work],
        'benefits': [benefits],
        'care_options': [care_options],
        'wellness_program': [wellness_program],
        'seek_help': [seek_help],
        'leave': [leave],
        'mental_health_consequence': [mental_health_consequence],
        'coworkers': [coworkers],
        'mental_health_interview': [mental_health_interview],
        'supervisor': [supervisor]
    })

# ------------------------
# Input for Treatment Prediction
# ------------------------
def inputs_for_treatment_prediction():
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    self_employed = st.selectbox('Are you self-employed?', ['Unknown', 'Yes', 'No'])
    family_history = st.selectbox("Do you have a family history of mental illness?", ['Yes', 'No'])
    work_interfere = st.selectbox(
        'If you have a mental health condition, do you feel that it interferes with your work?',
        ['Often', 'Rarely', 'Never', 'Sometimes', 'Unknown']
    )
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
    mental_health_interview = st.selectbox('Would you bring up a mental health issue in an interview?', ['No', 'Yes'])
    supervisor = st.selectbox('Would you discuss a mental health issue with your supervisor(s)?',
                              ['No', 'Maybe', 'Yes'])

    return pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'self_employed': [self_employed],
        'family_history': [family_history],
        'work_interfere': [work_interfere],
        'remote_work': [remote_work],
        'benefits': [benefits],
        'care_options': [care_options],
        'wellness_program': [wellness_program],
        'seek_help': [seek_help],
        'leave': [leave],
        'mental_health_consequence': [mental_health_consequence],
        'coworkers': [coworkers],
        'mental_health_interview': [mental_health_interview],
        'supervisor': [supervisor]
    })

# ------------------------
# About Section
# ------------------------
if section == "About":
    st.title("Mental Health Prediction & Clustering App")
    st.write("""
    This app demonstrates:
    - **Classification**: Predicting whether a person will seek mental health treatment.
    - **Regression**: Predicting the person's age based on survey responses.
    - **Unsupervised Clustering**: Grouping individuals into clusters based on similarities.
    """)

# ------------------------
# Age Prediction Section
# ------------------------
elif section == "Age Prediction":
    st.title("Age Prediction")
    input_df = inputs_for_age_prediction()
    if st.button("Predict Age"):
        prediction = age_model.predict(input_df)[0]
        st.success(f"Predicted Age: {prediction:.1f} years")

# ------------------------
# Treatment Prediction Section
# ------------------------
elif section == "Treatment Prediction":
    st.title("Treatment Prediction")
    input_df = inputs_for_treatment_prediction()
    if st.button("Predict Treatment"):
        prediction = treatment_model.predict(input_df)[0]
        st.success(f"Treatment Prediction: {'Yes' if prediction == 1 else 'No'}")

# ------------------------
# Clustering Report Section
# ------------------------
elif section == "Clustering Report":
    st.title("Clustering Analysis")
    st.image("Images/cluster_plot.png", caption="Cluster Visualization", use_column_width=True)
    st.write('''
   
 **Cluster 0: "Minimal Mental Health Awareness"**
- **Characteristics**: 
  - Employees in this cluster often work in large companies (e.g., `no_employees = 1001`).
  - They frequently respond with "Don't know" or "No" to questions about mental health benefits, wellness programs, and seeking help.
  - Low engagement with mental health resources and support systems.
  - Less likely to have sought treatment or discussed mental health issues openly.
- **Rationale**: This group shows limited awareness or engagement with mental health support, possibly due to workplace culture or lack of resources.

---

 **Cluster 1: "Moderate Engagement with Mental Health Support"**
- **Characteristics**: 
  - Mixed responses to mental health benefits and wellness programs (some "Yes," some "Don't know").
  - Moderate levels of seeking help or discussing mental health with coworkers/supervisors.
  - Somewhat aware of mental health resources but not fully utilizing them.
- **Rationale**: These employees are somewhat engaged with mental health support but may lack consistent access or confidence in workplace resources.

---

 **Cluster 2: "Proactive Mental Health Advocates"**
- **Characteristics**: 
  - Higher likelihood of responding "Yes" to mental health benefits, wellness programs, and seeking help.
  - More open about discussing mental health with coworkers/supervisors.
  - Often work in tech companies or smaller organizations where mental health support is more accessible.
  - More likely to have sought treatment or taken leave for mental health reasons.
- **Rationale**: This group actively engages with mental health resources and advocates for support in the workplace.

| Cluster | Name                              | Key Traits                                                                 |
|---------|-----------------------------------|---------------------------------------------------------------------------|
| 0       | Minimal Mental Health Awareness   | Low engagement, large companies, "Don't know" responses.                  |
| 1       | Moderate Engagement with Support  | Mixed responses, some awareness but inconsistent utilization.             |
| 2       | Proactive Mental Health Advocates | High engagement, open discussions, likely to seek treatment.              |

''')
    st.image("Images/cluster_report.png",  use_column_width=True)
