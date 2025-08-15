import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set page config
st.set_page_config(page_title="Mental Health Analysis", page_icon="🧠", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .st-bb {
        background-color: #ffffff;
    }
    .st-at {
        background-color: #e6f3ff;
    }
    .st-ae {
        background-color: #ffffff;
    }
    .cluster-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4682b4;
    }
    .prediction-box {
        background-color: #f0fff0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        font-size: 18px;
        border-left: 5px solid #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox("Go to", ["About", "Predict Age", "Treatment Seeking Employees", "Clustering Report"])

# Load data (cached)
@st.cache_data
def load_data():
    return pd.read_csv('clean_data.csv')

df = load_data()

# About page
if page == 'About':
    st.header('🧠 Mental Health in Tech Industry Survey Analysis')
    st.markdown("""
    This interactive dashboard analyzes data from a 2014 survey about mental health attitudes 
    and frequency of mental health disorders in the tech workplace.
    """)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('[Dataset Source: Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)')
        st.write("The dataset contains responses from tech industry employees about their mental health experiences, workplace support, and personal attitudes.")
    
    with col2:
        st.image('Images/Skewness.png', width=200)
    
    st.subheader('Key Insights')
    st.markdown("""
    - **Skewness in Age**: The dataset shows a right-skewed age distribution (skewness = 1.05)
    - **Treatment Seeking**: About 50% of respondents have sought treatment for mental health
    - **Work Impact**: 35% report mental health sometimes interferes with work
    """)
    
    st.subheader('Dataset Preview')
    st.dataframe(df.head())
    
    st.subheader('Data Distribution')
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        df['Gender'].value_counts().head(3).plot(kind='bar', ax=ax)
        ax.set_title('Gender Distribution')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        df['treatment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_title('Treatment Seeking')
        ax.set_ylabel('')
        st.pyplot(fig)

# Predict Age page
elif page == 'Predict Age':
    st.header("📊 Age Prediction")
    st.markdown("""
    Predict the age of an employee based on their mental health survey responses.
    *Note: This is for demonstration purposes only as the dataset isn't ideal for age prediction.*
    """)
    
    with st.expander("ℹ️ About this Model"):
        st.write("""
        - **Model Type**: Linear Regression
        - **Features Used**: 15 survey responses about mental health attitudes
        - **Limitations**: Age prediction from survey responses is inherently challenging
        """)
    
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
    
    with st.expander("Additional Survey Responses"):
        col1, col2 = st.columns(2)
        with col1:
            care_options = st.selectbox('Know mental health care options?', ['Not sure', 'No', 'Yes'])
            wellness_program = st.selectbox('Employer discusses mental health?', ["Don't know", 'Yes', 'No'])
            seek_help = st.selectbox('Resources for mental health help?', ['Yes', 'No'])
        
        with col2:
            leave = st.selectbox('Ease of taking mental health leave?',
                               ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult', "Don't know"])
            mental_health_consequence = st.selectbox('Negative consequences for discussing?', ['No', 'Maybe', 'Yes'])
    
    # Create input dataframe
    input_df = pd.DataFrame([{
        'Gender': gender,
        'self_employed': 'Unknown',
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
        'coworkers': 'Some of them',
        'mental_health_interview': 'No',
        'supervisor': 'No'
    }])
    
    if st.button('Predict Age', type='primary'):
        try:
            model = joblib.load('Streamlit/reg_model.pkl')
            predicted_age = model.predict(input_df)
            predicted_age = np.expm1(predicted_age)[0]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Age</h3>
                <h2>{predicted_age:.1f} years</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Show actual age distribution for context
            st.subheader("Age Distribution in Dataset")
            fig, ax = plt.subplots()
            sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=ax)
            ax.axvline(predicted_age, color='red', linestyle='--', label='Predicted Age')
            ax.set_title('Age Distribution with Prediction')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Treatment Seeking page
elif page == "Treatment Seeking Employees":
    st.header("🧠 Mental Health Treatment Prediction")
    st.markdown("""
    Predict whether an employee is likely to seek mental health treatment based on their survey responses.
    *Model: Random Forest Classifier (accuracy: 85%)*
    """)
    
    with st.expander("Model Details"):
        st.write("""
        - **Algorithm**: Random Forest Classifier
        - **Features**: 15 survey responses
        - **Target**: Whether employee sought treatment (binary)
        - **Evaluation Metrics**:
            - Accuracy: 85%
            - Precision: 86%
            - Recall: 92%
        """)
    
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
        mental_health_consequence = st.selectbox('Negative consequences for discussing?', ['No', 'Maybe', 'Yes'])
    
    # Create input dataframe
    input_df = pd.DataFrame([{
        'Gender': gender,
        'self_employed': 'Unknown',
        'family_history': family_history,
        'treatment': 'No',
        'work_interfere': work_interfere,
        'remote_work': 'No',
        'benefits': benefits,
        'care_options': 'Not sure',
        'wellness_program': wellness_program,
        'seek_help': 'No',
        'leave': 'Somewhat easy',
        'mental_health_consequence': mental_health_consequence,
        'coworkers': 'Some of them',
        'mental_health_interview': 'No',
        'supervisor': 'No'
    }])
    
    if st.button('Predict Treatment Seeking', type='primary'):
        try:
            clf = joblib.load('Streamlit/clf_model.pkl')
            predicted_treatment = clf.predict(input_df)[0]
            proba = clf.predict_proba(input_df)[0][1]
            
            if predicted_treatment == 1:
                result = "Likely to seek treatment"
                color = "#4CAF50"
            else:
                result = "Unlikely to seek treatment"
                color = "#F44336"
            
            st.markdown(f"""
            <div class="prediction-box" style="border-left-color: {color}">
                <h3>Prediction Result</h3>
                <h2 style="color: {color}">{result}</h2>
                <p>Confidence: {proba*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show feature importance if available
            try:
                st.subheader("Key Factors in Prediction")
                feature_importance = pd.DataFrame({
                    'Feature': input_df.columns,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False).head(5)
                
                fig, ax = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='Blues_d', ax=ax)
                ax.set_title('Top 5 Important Features')
                st.pyplot(fig)
            except:
                pass
            
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Clustering Report page
elif page == 'Clustering Report':
    st.header("👥 Employee Clustering Analysis")
    st.markdown("""
    Employees clustered based on their attitudes toward mental health using K-Means with dimensionality reduction.
    """)
    
    st.image('Images/Clusters.png', use_column_width=True)
    
    st.subheader("Cluster Characteristics")
    
    st.markdown("""
    <div class="cluster-box">
        <h3>Cluster 0: Supervisor-Reliant Onsite Workers</h3>
        <ul>
            <li>Family history: Very low (2%)</li>
            <li>Treatment: High (90%)</li>
            <li>Work interference: Moderate (47%)</li>
            <li>Remote work: None</li>
            <li>Communication: Open to supervisors (58%), moderate to coworkers/employers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="cluster-box">
        <h3>Cluster 1: Treated but Employer-Wary</h3>
        <ul>
            <li>Family history: Very low (2%)</li>
            <li>Treatment: Very high (93%)</li>
            <li>Work interference: High (60%)</li>
            <li>Remote work: None</li>
            <li>Communication: Moderate with coworkers and supervisors, low with employers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="cluster-box">
        <h3>Cluster 2: Remote High-Risk Communicators</h3>
        <ul>
            <li>Family history: Very low (1%)</li>
            <li>Treatment: Moderate (74%)</li>
            <li>Work interference: Moderate-High (52%)</li>
            <li>Remote work: Fully remote</li>
            <li>Communication: Open across all levels</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Business Recommendations")
    st.markdown("""
    1. **For Cluster 0**: Focus on supervisor training programs to better support employees
    2. **For Cluster 1**: Improve employer communication channels to reduce wariness
    3. **For Cluster 2**: Develop remote-specific mental health resources and support systems
    """)
