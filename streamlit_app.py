import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CODVEDA Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 CODVEDA Technologies - Data Analysis Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Home", "Iris Regression", "Iris Classification", "Sentiment Analysis"]
)

# Home Page
if analysis_type == "Home":
    st.header("Welcome to CODVEDA Data Analysis Platform")
    st.write("""
    This application provides three types of analysis:
    
    1. **Iris Regression Analysis** - Predicting petal and sepal measurements
    2. **Iris Classification** - Multi-model classification with Random Forest, Decision Tree, and Logistic Regression
    3. **Sentiment Analysis** - NLP-based sentiment classification
    
    Use the sidebar to navigate between different analyses.
    """)

# Iris Regression Analysis
elif analysis_type == "Iris Regression":
    st.header("🌸 Iris Dataset - Linear Regression Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Iris CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Species Count", df['species'].nunique())
        
        st.dataframe(df.head())
        
        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Correlation heatmap
        st.subheader("Feature Correlation Matrix")
        numerical_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Regression Model
        st.subheader("🎯 Regression Model: Predicting Petal Length")
        
        X = df[['sepal_length', 'sepal_width', 'petal_width']]
        y = df['petal_length']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
        
        # Actual vs Predicted plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Petal Length')
        ax.set_ylabel('Predicted Petal Length')
        ax.set_title('Actual vs Predicted Values')
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        st.dataframe(coef_df)

# Iris Classification
elif analysis_type == "Iris Classification":
    st.header("🌺 Iris Classification - Multi-Model Comparison")
    
    uploaded_file = st.file_uploader("Upload Iris CSV file", type=['csv'])
    
    if uploaded_file is not None:
        from sklearn.datasets import load_iris
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        
        iris_data = load_iris()
        df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        df['species'] = iris_data.target
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Prepare data
        X = df.drop('species', axis=1)
        y = df['species']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = []
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append({'Model': name, 'Accuracy': accuracy})
        
        # Display results
        st.subheader("Model Comparison")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.highlight_max(subset=['Accuracy']))
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results_df['Model'], results_df['Accuracy'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim([0.8, 1.0])
        st.pyplot(fig)
        
        # Best model details
        best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        st.success(f"🏆 Best Model: {best_model}")

# Sentiment Analysis
elif analysis_type == "Sentiment Analysis":
    st.header("💬 Sentiment Analysis - NLP")
    
    uploaded_file = st.file_uploader("Upload Sentiment CSV file", type=['csv'])
    
    if uploaded_file is not None:
        import re
        from textblob import TextBlob
        from wordcloud import WordCloud
        
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Text preprocessing
        def preprocess_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text
        
        df['cleaned_text'] = df['Text'].apply(preprocess_text)
        
        # Sentiment analysis
        def get_sentiment(text):
            analysis = TextBlob(text)
            if analysis.polarity > 0:
                return 'Positive'
            elif analysis.polarity < 0:
                return 'Negative'
            else:
                return 'Neutral'
        
        df['sentiment'] = df['cleaned_text'].apply(get_sentiment)
        
        # Display sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", sentiment_counts.get('Positive', 0))
        with col2:
            st.metric("Neutral", sentiment_counts.get('Neutral', 0))
        with col3:
            st.metric("Negative", sentiment_counts.get('Negative', 0))
        
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
        
        # Word cloud
        st.subheader("Word Cloud")
        text = " ".join(df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**CODVEDA Technologies** | Data Analysis Platform")
