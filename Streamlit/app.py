import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image 

# Set page config
st.set_page_config(page_title="Mental Health Analysis", layout="wide")
age_labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
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
            
          # Get model prediction and transform if needed
            raw_prediction = model.predict(input_df)[0]
            predicted_value = np.expm1(raw_prediction)  # Only use if you did log1p transformation during training

# Convert to age group
            age_group_index = int(np.round(predicted_value))
            age_group_index = np.clip(age_group_index, 0, len(age_labels)-1)  # Ensure valid index
            predicted_age_group = age_labels[age_group_index]

# Get age range for visualization
            age_range_start = age_bins[age_group_index]
            age_range_end = age_bins[age_group_index+1]
            age_midpoint = (age_range_start + age_range_end) / 2

# Create visualization
            st.subheader("Age Distribution Analysis")

            fig, ax = plt.subplots(figsize=(10, 6))

# Plot age distribution
            sns.histplot(
            df['Age'].dropna(), 
            bins=age_bins, 
            kde=False, 
            ax=ax, 
            color='skyblue',
            alpha=0.7
            )

# Add prediction indicator
            ax.axvline(
            age_midpoint, 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'Predicted: {predicted_age_group}'
            )
    
# Add reference lines for age group boundaries
            ax.axvline(age_range_start, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(age_range_end, color='gray', linestyle=':', alpha=0.5)

# Format plot
            ax.set_title('Age Distribution with Prediction', pad=20)
            ax.set_xlabel('Age (years)')
            ax.set_ylabel('Number of People')
            ax.legend()

# Display plot
            st.pyplot(fig)

# Add explanatory text
            st.markdown(f"""
            ### Prediction Details
            - **Predicted Age Group**: {predicted_age_group} (ages {age_range_start}-{age_range_end})
            - **Midpoint Age**: {age_midpoint:.1f} years
            - **Raw Model Output**: {raw_prediction:.2f}  
                      {'(Exponentiated: ' + str(round(predicted_value, 2)) if 'predicted_value' in locals() else ''}

            The histogram shows the age distribution in our dataset, with the red dashed line indicating 
            the midpoint of your predicted age group. Gray dotted lines show the boundaries of this age group.
            """)
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
    st.image(r'Images/clusters_plot.png')
    st.subheader("Cluster Characteristics")
    st.image(r'Images/clusters012.png')
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
    st.image(r'Images/clusters_report.png')
     # For loading PNG files

# Add clustering report section
    st.header("Employee Mental Health Clusters Analysis")

# Create columns for cluster summaries
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Cluster 0")
        st.markdown("""
        **Supervisor-Reliant Onsite Workers**  
        - Family history: Very low (2%)  
        - Treatment: High (90%)  
        - Work interference: Moderate (47%)  
        """)

    with col2:
        st.subheader("Cluster 1")
        st.markdown("""
        **Treated but Employer-Wary**  
        - Family history: Very low (2%)  
        - Treatment: Very high (93%)  
        - Work interference: High (60%)  
        """)

    with col3:
        st.subheader("Cluster 2")
        st.markdown("""
        **Remote High-Risk Communicators**  
        - Family history: Very low (1%)  
         - Treatment: Moderate (74%)  
        - Work interference: Moderate-High (52%)  
        """)

# Add cluster visualizations
    st.subheader("Cluster Characteristics")

# Assuming you have these PNG files in your directory
    cluster_images = {
        "Feature Correlations": "Images/famhisvsclusters.png",
        "Demographic Breakdown": "Images/gendervsclusters.png",
        "Treatment Patterns": "Images/treatmentvsclusters.png"
    }

# Display images with captions
    for description, img_path in cluster_images.items():
        try:
            image = Image.open(img_path)
            st.image(image, caption=description, use_column_width=True)
        except FileNotFoundError:
            st.warning(f"Image not found: {img_path}")

# Add interpretation section
    st.subheader("Key Insights")
    st.markdown("""
    1. **Cluster 0** shows high treatment rates despite low family history, suggesting workplace factors dominate  
    2. **Cluster 1** has the highest work interference, indicating severe productivity impacts  
    3. **Cluster 2** represents remote workers needing communication-focused interventions  
    """)

# Optional: Add expandable technical details
    with st.expander("Methodology Details"):
        st.markdown("""
        - Clustering performed using K-Means (k=3)  
        - Features scaled using StandardScaler  
        - Optimal k determined via elbow method  
        - Silhouette score: 0.39  
        """)
