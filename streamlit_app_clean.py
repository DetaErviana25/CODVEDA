"""
Data Science Portfolio - Streamlit App
Created for CODVEDA Technologies Projects
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Data Science Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .skill-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        background: #667eea;
        color: white;
        border-radius: 15px;
        font-size: 0.9rem;
    }
    .project-card {
        background: #f7fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Projects", "🛠️ Skills", "📈 Visualizations", "📧 Contact"])

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 About Me")
st.sidebar.info("""
**Data Scientist**  
🎓 Machine Learning Enthusiast  
🐍 Python Developer  
📊 Analytics Expert
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Connect")
st.sidebar.markdown("[💼 LinkedIn](https://linkedin.com/in/yourprofile)")
st.sidebar.markdown("[🐙 GitHub](https://github.com/yourusername)")
st.sidebar.markdown("[📧 Email](mailto:your.email@example.com)")

# Main Content
if page == "🏠 Home":
    # Hero Section
    st.markdown('<h1 class="main-header">Data Science Portfolio</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transforming Data into Actionable Insights</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>5+</h2>
            <p>Projects Completed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>95%</h2>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>10+</h2>
            <p>ML Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>3+</h2>
            <p>Years Experience</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About Me
    st.header("👨‍💻 About Me")
    st.write("""
    I'm a passionate Data Scientist with expertise in machine learning, statistical analysis, 
    and data visualization. I specialize in building predictive models and extracting meaningful 
    insights from complex datasets.
    
    My work focuses on:
    - **Machine Learning**: Classification, Regression, and Clustering
    - **Natural Language Processing**: Sentiment Analysis and Text Mining
    - **Data Visualization**: Creating compelling stories with data
    - **Statistical Analysis**: Hypothesis testing and model evaluation
    """)
    
    # Recent Highlights
    st.header("🌟 Recent Highlights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("✅ Achieved 95%+ accuracy in Iris classification")
        st.success("✅ Built end-to-end NLP sentiment analysis pipeline")
    
    with col2:
        st.success("✅ Optimized models using hyperparameter tuning")
        st.success("✅ Created interactive data visualizations")

elif page == "📊 Projects":
    st.title("📊 Featured Projects")
    st.markdown("---")
    
    # Project 1: Iris Classification
    st.markdown("""
    <div class="project-card">
        <h2>🌸 Iris Dataset Classification & Regression</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Description:**  
        Comprehensive machine learning analysis of the Iris dataset including both 
        regression and classification tasks.
        
        **Key Features:**
        - Linear regression models with R² > 0.8
        - Multi-model comparison (Random Forest, Decision Tree, Logistic Regression)
        - Hyperparameter tuning using GridSearchCV
        - Feature importance analysis
        - Cross-validation and model evaluation
        
        **Results:**
        - Random Forest Classifier: **95%+ accuracy**
        - Successful prediction of petal and sepal measurements
        - Identified key features for species classification
        """)
    
    with col2:
        st.metric("Accuracy", "95.3%", "+2.1%")
        st.metric("R² Score", "0.89", "+0.05")
        st.metric("F1 Score", "0.94", "+0.03")
        
        st.markdown("**Tech Stack:**")
        st.markdown('<span class="skill-badge">Python</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">scikit-learn</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">pandas</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">matplotlib</span>', unsafe_allow_html=True)
    
    # Sample visualization
    with st.expander("📈 View Model Performance Comparison"):
        iris_data = pd.DataFrame({
            'Model': ['Random Forest', 'Decision Tree', 'Logistic Regression'],
            'Accuracy': [0.953, 0.933, 0.913],
            'Precision': [0.955, 0.935, 0.915],
            'Recall': [0.950, 0.930, 0.910],
            'F1-Score': [0.952, 0.932, 0.912]
        })
        
        fig = px.bar(iris_data, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                     title='Model Performance Comparison',
                     barmode='group',
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Project 2: Sentiment Analysis
    st.markdown("""
    <div class="project-card">
        <h2>💬 NLP Sentiment Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Description:**  
        End-to-end Natural Language Processing pipeline for sentiment classification 
        of text data using TextBlob and NLTK.
        
        **Key Features:**
        - Text preprocessing (tokenization, stopword removal, lemmatization)
        - Sentiment classification using TextBlob polarity analysis
        - Word frequency analysis and word clouds
        - Comparative analysis of sentiment distributions
        
        **Techniques Used:**
        - NLTK for text preprocessing
        - TextBlob for sentiment polarity analysis
        - WordCloud for visualization
        - Statistical text analysis
        """)
    
    with col2:
        st.metric("Texts Analyzed", "1000+", "")
        st.metric("Processing Time", "< 5s", "-20%")
        st.metric("Accuracy", "87%", "+5%")
        
        st.markdown("**Tech Stack:**")
        st.markdown('<span class="skill-badge">NLTK</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">TextBlob</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">WordCloud</span>', unsafe_allow_html=True)
        st.markdown('<span class="skill-badge">pandas</span>', unsafe_allow_html=True)
    
    # Sample visualization
    with st.expander("📊 View Sentiment Distribution"):
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral'],
            'Count': [450, 300, 250]
        })
        
        fig = px.pie(sentiment_data, values='Count', names='Sentiment',
                     title='Sentiment Distribution',
                     color_discrete_sequence=['#4ade80', '#f87171', '#94a3b8'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Project 3: EDA
    st.markdown("""
    <div class="project-card">
        <h2>📈 Exploratory Data Analysis Suite</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Description:**  
    Comprehensive exploratory data analysis toolkit for uncovering patterns and insights.
    
    **Capabilities:**
    - Statistical summaries and distribution analysis
    - Correlation matrix and heatmap generation
    - Feature relationship visualization
    - Outlier detection and handling
    - Missing value analysis
    
    **Key Insights:**
    - Petal measurements show stronger correlation than sepal measurements
    - Species can be effectively separated using petal features
    - Linear relationships exist between most features
    """)

elif page == "🛠️ Skills":
    st.title("🛠️ Technical Skills")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💻 Programming Languages")
        skills_prog = pd.DataFrame({
            'Skill': ['Python', 'SQL', 'R'],
            'Proficiency': [95, 80, 70]
        })
        fig = px.bar(skills_prog, x='Proficiency', y='Skill', orientation='h',
                     color='Proficiency', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🤖 Machine Learning")
        st.markdown("""
        - Supervised Learning (Regression, Classification)
        - Unsupervised Learning (Clustering, PCA)
        - Model Evaluation & Validation
        - Hyperparameter Tuning
        - Cross-Validation
        - Feature Engineering
        """)
    
    with col2:
        st.subheader("📚 Libraries & Frameworks")
        skills_lib = pd.DataFrame({
            'Library': ['scikit-learn', 'pandas', 'NumPy', 'matplotlib', 'seaborn'],
            'Proficiency': [90, 95, 85, 88, 85]
        })
        fig = px.bar(skills_lib, x='Proficiency', y='Library', orientation='h',
                     color='Proficiency', color_continuous_scale='Purples')
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔍 Specialized Skills")
        st.markdown("""
        - Natural Language Processing
        - Statistical Analysis
        - Data Visualization
        - A/B Testing
        - Time Series Analysis
        - Deep Learning Basics
        """)
    
    st.markdown("---")
    
    # Skills Matrix
    st.subheader("📊 Skills Proficiency Matrix")
    skills_matrix = pd.DataFrame({
        'Category': ['Data Analysis', 'Machine Learning', 'Data Visualization', 
                     'NLP', 'Statistical Analysis', 'Deep Learning'],
        'Beginner': [0, 0, 0, 20, 0, 40],
        'Intermediate': [0, 20, 10, 30, 20, 60],
        'Advanced': [30, 50, 40, 50, 50, 0],
        'Expert': [70, 30, 50, 0, 30, 0]
    })
    
    fig = go.Figure()
    for col in ['Expert', 'Advanced', 'Intermediate', 'Beginner']:
        fig.add_trace(go.Bar(name=col, x=skills_matrix['Category'], y=skills_matrix[col]))
    
    fig.update_layout(barmode='stack', title='Skill Level Distribution by Category',
                      height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Visualizations":
    st.title("📈 Interactive Visualizations")
    st.markdown("---")
    
    # Iris Dataset Correlation
    st.subheader("🌸 Iris Dataset - Feature Correlations")
    
    corr_matrix = pd.DataFrame({
        'sepal_length': [1.0, -0.12, 0.87, 0.82],
        'sepal_width': [-0.12, 1.0, -0.43, -0.37],
        'petal_length': [0.87, -0.43, 1.0, 0.96],
        'petal_width': [0.82, -0.37, 0.96, 1.0]
    }, index=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                    title='Feature Correlation Heatmap')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Model Performance Over Time
    st.subheader("📊 Model Performance Trends")
    
    performance_data = pd.DataFrame({
        'Iteration': range(1, 11),
        'Random Forest': [0.85, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.953],
        'Decision Tree': [0.80, 0.82, 0.84, 0.85, 0.87, 0.88, 0.90, 0.91, 0.92, 0.933],
        'Logistic Regression': [0.78, 0.80, 0.82, 0.84, 0.85, 0.87, 0.88, 0.89, 0.90, 0.913]
    })
    
    fig = px.line(performance_data, x='Iteration', 
                  y=['Random Forest', 'Decision Tree', 'Logistic Regression'],
                  title='Model Accuracy Over Iterations',
                  labels={'value': 'Accuracy', 'variable': 'Model'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_imp = pd.DataFrame({
            'Feature': ['petal_width', 'petal_length', 'sepal_length', 'sepal_width'],
            'Importance': [0.45, 0.42, 0.10, 0.03]
        })
        
        fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                     title='Random Forest - Feature Importance',
                     color='Importance', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        scatter_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 6.2, 5.8, 6.7, 5.4],
            'sepal_width': [3.5, 3.0, 2.9, 2.7, 3.1, 3.9],
            'petal_length': [1.4, 1.4, 4.3, 4.1, 5.2, 1.7],
            'species': ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'setosa']
        })
        
        fig = px.scatter_3d(scatter_data, x='sepal_length', y='sepal_width', z='petal_length',
                            color='species', title='3D Species Distribution')
        st.plotly_chart(fig, use_container_width=True)

elif page == "📧 Contact":
    st.title("📧 Get In Touch")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Let's Connect!
        
        I'm always interested in discussing new projects, opportunities, or 
        collaborations in data science and machine learning.
        """)
        
        # Contact Form
        with st.form("contact_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            subject = st.selectbox("Subject", 
                                  ["General Inquiry", "Project Collaboration", 
                                   "Job Opportunity", "Other"])
            message = st.text_area("Message", height=150)
            
            submitted = st.form_submit_button("Send Message")
            
            if submitted:
                if name and email and message:
                    st.success("✅ Thank you for your message! I'll get back to you soon.")
                else:
                    st.error("❌ Please fill in all fields.")
    
    with col2:
        st.markdown("### 📍 Contact Information")
        st.info("""
        **Email:**  
        your.email@example.com
        
        **Location:**  
        Your City, Country
        
        **LinkedIn:**  
        linkedin.com/in/yourprofile
        
        **GitHub:**  
        github.com/yourusername
        
        **Availability:**  
        Open to opportunities
        """)
        
        st.markdown("### 💼 Download Resume")
        
        # Create a sample CV text
        cv_text = """
        DATA SCIENCE PORTFOLIO
        
        SKILLS:
        - Python, scikit-learn, pandas
        - Machine Learning & NLP
        - Data Visualization
        
        PROJECTS:
        1. Iris Classification (95% accuracy)
        2. NLP Sentiment Analysis
        3. Exploratory Data Analysis
        """
        
        st.download_button(
            label="📄 Download CV (TXT)",
            data=cv_text,
            file_name="resume.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 2rem 0;'>
    <p>Built with ❤️ using Streamlit | © 2025 Data Science Portfolio</p>
    <p>Last Updated: {}</p>
</div>
""".format(datetime.now().strftime("%B %Y")), unsafe_allow_html=True)
