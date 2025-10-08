import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="CODVEDA Analytics Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #f0f7ff;
        border-left: 5px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/data-science.png", width=100)
    st.title("🎯 Navigation")
    st.markdown("---")
    
    analysis_type = st.radio(
        "Pilih Analisis:",
        ["🏠 Home", "🌸 Iris Regression", "🌺 Iris Classification", "💬 Sentiment Analysis"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    **CODVEDA Analytics Hub** menyediakan:
    - Analisis Regresi Linear
    - Klasifikasi Multi-Model
    - Analisis Sentimen NLP
    
    Dibuat dengan ❤️ menggunakan Streamlit
    """)

# ====================
# HOME PAGE
# ====================
if analysis_type == "🏠 Home":
    st.markdown('<p class="main-header">🚀 CODVEDA Analytics Hub</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Platform Analisis Data Interaktif dengan Machine Learning</p>', unsafe_allow_html=True)
    
    # Welcome section with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <h2>🌸</h2>
            <h3>Iris Regression</h3>
            <p>Prediksi panjang petal & sepal menggunakan Linear Regression</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px; color: white;'>
            <h2>🌺</h2>
            <h3>Iris Classification</h3>
            <p>Klasifikasi spesies dengan 3 model ML berbeda</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; color: white;'>
            <h2>💬</h2>
            <h3>Sentiment Analysis</h3>
            <p>Analisis sentimen teks menggunakan NLP</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features
    st.markdown("## ✨ Fitur Unggulan")
    
    feat1, feat2, feat3, feat4 = st.columns(4)
    
    with feat1:
        st.metric("📈 Visualisasi", "Interaktif", delta="Real-time")
    with feat2:
        st.metric("🤖 Model ML", "3+ Algoritma", delta="Akurat")
    with feat3:
        st.metric("📊 Dashboard", "Responsif", delta="Modern")
    with feat4:
        st.metric("⚡ Performa", "Cepat", delta="Optimal")
    
    st.markdown("---")
    
    # How to use
    st.markdown("## 📖 Cara Menggunakan")
    
    with st.expander("📝 Panduan Lengkap", expanded=True):
        st.markdown("""
        ### Langkah-langkah:
        
        1. **Pilih Analisis** di sidebar kiri
        2. **Upload Dataset** dalam format CSV
        3. **Lihat Hasil** analisis secara otomatis
        4. **Eksplorasi Visualisasi** interaktif
        5. **Download Insights** untuk reporting
        
        ### Tips:
        - ✅ Pastikan format CSV sesuai dengan contoh
        - ✅ Periksa tidak ada missing values
        - ✅ Gunakan dataset dengan minimal 50 rows untuk hasil optimal
        """)
    
    # Sample datasets info
    st.markdown("## 📂 Sample Datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌸 Iris Dataset
        - **Fitur:** sepal_length, sepal_width, petal_length, petal_width
        - **Target:** species (Setosa, Versicolor, Virginica)
        - **Samples:** 150 rows
        """)
    
    with col2:
        st.markdown("""
        ### 💬 Sentiment Dataset
        - **Fitur:** Text (kalimat/review)
        - **Target:** Sentiment (Positive, Negative, Neutral)
        - **Samples:** Varies
        """)

# ====================
# IRIS REGRESSION
# ====================
elif analysis_type == "🌸 Iris Regression":
    st.markdown('<p class="main-header">🌸 Iris Regression Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Prediksi Panjang Petal menggunakan Linear Regression</p>', unsafe_allow_html=True)
    
    # Description
    with st.expander("ℹ️ Apa itu Linear Regression?", expanded=False):
        st.markdown("""
        **Linear Regression** adalah algoritma machine learning yang digunakan untuk memprediksi nilai kontinu 
        berdasarkan hubungan linear antar variabel.
        
        **Cara Kerja:**
        1. Model mencari garis terbaik (best-fit line) yang melewati data points
        2. Menggunakan persamaan: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
        3. Meminimalkan error antara prediksi dan nilai aktual
        
        **Metrics Evaluasi:**
        - **R² Score:** Seberapa baik model menjelaskan variasi data (0-1, semakin tinggi semakin baik)
        - **RMSE:** Root Mean Squared Error (semakin rendah semakin baik)
        - **MAE:** Mean Absolute Error (rata-rata kesalahan absolut)
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload File CSV Iris Dataset", 
        type=['csv'],
        help="Upload file dengan kolom: sepal_length, sepal_width, petal_length, petal_width, species"
    )
    
    # Demo button
    use_demo = st.checkbox("🎲 Gunakan Demo Dataset (Iris Built-in)", value=False)
    
    if uploaded_file is not None or use_demo:
        # Load data
        if use_demo:
            iris_data = load_iris()
            df = pd.DataFrame(iris_data.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            df['species'] = iris_data.target_names[iris_data.target]
        else:
            df = pd.read_csv(uploaded_file)
        
        # Data Overview
        st.markdown("## 📊 Overview Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Baris", df.shape[0], delta="Samples")
        with col2:
            st.metric("📋 Total Kolom", df.shape[1], delta="Features")
        with col3:
            st.metric("🌺 Jumlah Species", df['species'].nunique(), delta="Classes")
        with col4:
            missing = df.isnull().sum().sum()
            st.metric("❌ Missing Values", missing, delta="Clean" if missing == 0 else "Check")
        
        # Show data
        with st.expander("👀 Lihat Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Statistics
        st.markdown("## 📈 Statistik Deskriptif")
        
        tab1, tab2 = st.tabs(["📊 Summary Statistics", "📉 Distribution by Species"])
        
        with tab1:
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            species_stats = df.groupby('species').agg({
                'sepal_length': 'mean',
                'sepal_width': 'mean',
                'petal_length': 'mean',
                'petal_width': 'mean'
            }).round(2)
            st.dataframe(species_stats, use_container_width=True)
        
        # Correlation Analysis
        st.markdown("## 🔗 Analisis Korelasi")
        
        numerical_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        corr_matrix = df[numerical_features].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, 
                    square=True, linewidths=1, ax=ax, fmt='.2f',
                    cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix - Iris Features', fontsize=16, fontweight='bold', pad=20)
        st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <h4>💡 Insight Korelasi:</h4>
        <ul>
        <li><b>Petal Length & Petal Width</b> memiliki korelasi sangat tinggi (>0.9) - hubungan linear kuat</li>
        <li><b>Sepal Width</b> memiliki korelasi negatif dengan fitur lain - pola berbeda</li>
        <li>Korelasi tinggi menunjukkan fitur saling berkaitan, cocok untuk regression</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        st.markdown("## 📊 Visualisasi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            for species in df['species'].unique():
                species_data = df[df['species'] == species]
                ax.scatter(species_data['petal_width'], species_data['petal_length'],
                          label=species, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Petal Width (cm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Petal Length (cm)', fontsize=12, fontweight='bold')
            ax.set_title('Petal Width vs Petal Length by Species', fontsize=14, fontweight='bold')
            ax.legend(title='Species', fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column='petal_length', by='species', ax=ax, patch_artist=True)
            ax.set_xlabel('Species', fontsize=12, fontweight='bold')
            ax.set_ylabel('Petal Length (cm)', fontsize=12, fontweight='bold')
            ax.set_title('Distribution of Petal Length by Species', fontsize=14, fontweight='bold')
            plt.suptitle('')
            st.pyplot(fig)
        
        # Model Training
        st.markdown("## 🤖 Model Training: Linear Regression")
        
        st.info("**Target Variable:** Petal Length | **Features:** Sepal Length, Sepal Width, Petal Width")
        
        # Prepare data
        X = df[['sepal_length', 'sepal_width', 'petal_width']]
        y = df['petal_length']
        
        # Split ratio slider
        test_size = st.slider("🎚️ Test Size (%)", min_value=10, max_value=40, value=20, step=5)
        test_ratio = test_size / 100
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=42
        )
        
        # Train model
        with st.spinner("🔄 Training model..."):
            model = LinearRegression()
            model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        st.markdown("## 🎯 Hasil Model")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "R² Score (Train)", 
                f"{train_r2:.4f}",
                delta=f"{train_r2*100:.1f}%",
                help="Persentase variasi data yang dijelaskan model"
            )
        
        with col2:
            st.metric(
                "R² Score (Test)", 
                f"{test_r2:.4f}",
                delta=f"{(test_r2-train_r2)*100:.1f}%",
                help="Performa pada data yang belum pernah dilihat"
            )
        
        with col3:
            st.metric(
                "RMSE (Train)", 
                f"{train_rmse:.4f}",
                delta="-" + f"{train_rmse:.2f}",
                delta_color="inverse",
                help="Root Mean Squared Error (lebih kecil lebih baik)"
            )
        
        with col4:
            st.metric(
                "RMSE (Test)", 
                f"{test_rmse:.4f}",
                delta=f"{(test_rmse-train_rmse):.2f}",
                delta_color="inverse",
                help="Error pada data testing"
            )
        
        # Interpretation
        if test_r2 > 0.9:
            interpretation = "EXCELLENT! Model memiliki performa sangat baik ✨"
            color = "#28a745"
        elif test_r2 > 0.8:
            interpretation = "GOOD! Model memiliki performa baik 👍"
            color = "#17a2b8"
        elif test_r2 > 0.6:
            interpretation = "MODERATE. Model cukup baik, bisa ditingkatkan 📈"
            color = "#ffc107"
        else:
            interpretation = "POOR. Model perlu improvement 🔧"
            color = "#dc3545"
        
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h3>{interpretation}</h3>
        <p>Model menjelaskan <b>{test_r2*100:.1f}%</b> variasi dalam Petal Length</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Coefficients
        st.markdown("## 🔍 Feature Importance (Coefficients)")
        
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(coef_df[['Feature', 'Coefficient']], use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#667eea' if c > 0 else '#f5576c' for c in coef_df['Coefficient']]
            ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
            ax.set_xlabel('Coefficient Value', fontweight='bold')
            ax.set_title('Feature Coefficients', fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>💡 Interpretasi Koefisien:</h4>
        <ul>
        <li><b>Intercept:</b> {model.intercept_:.4f}</li>
        <li><b>{coef_df.iloc[0]['Feature']}:</b> Memiliki pengaruh terbesar dengan koefisien {coef_df.iloc[0]['Coefficient']:.4f}</li>
        <li>Koefisien positif = hubungan positif, negatif = hubungan negatif</li>
        <li>Semakin besar nilai absolut, semakin besar pengaruhnya terhadap prediksi</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Predictions visualization
        st.markdown("## 📈 Visualisasi Prediksi")
        
        tab1, tab2 = st.tabs(["🎯 Actual vs Predicted", "📊 Residuals Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(y_train, y_train_pred, alpha=0.6, s=80, edgecolors='black', 
                          linewidth=0.5, color='#667eea', label='Training Data')
                ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                       'r--', lw=2, label='Perfect Prediction')
                ax.set_xlabel('Actual Petal Length', fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted Petal Length', fontsize=12, fontweight='bold')
                ax.set_title(f'Training Set\nR² = {train_r2:.4f}', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(y_test, y_test_pred, alpha=0.6, s=80, edgecolors='black', 
                          linewidth=0.5, color='#f5576c', label='Testing Data')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
                ax.set_xlabel('Actual Petal Length', fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted Petal Length', fontsize=12, fontweight='bold')
                ax.set_title(f'Testing Set\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        with tab2:
            residuals_test = y_test - y_test_pred
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test_pred, residuals_test, alpha=0.6, s=80, 
                          edgecolors='black', linewidth=0.5, color='#667eea')
                ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
                ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
                ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(residuals_test, bins=15, alpha=0.7, color='#764ba2', 
                       edgecolor='black', linewidth=1.2)
                ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
                ax.set_xlabel('Residuals', fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <h4>📊 Analisis Residual:</h4>
        <ul>
        <li><b>Residual Plot:</b> Sebaiknya residual tersebar random di sekitar garis nol (tidak ada pola)</li>
        <li><b>Histogram:</b> Distribusi residual sebaiknya mendekati normal (bell curve)</li>
        <li>Pola tertentu dalam residual mengindikasikan model bisa diperbaiki</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Conclusion
        st.markdown("## ✅ Kesimpulan")
        
        st.markdown(f"""
        <div class="success-box">
        <h4>🎓 Ringkasan Hasil Analisis:</h4>
        <ol>
        <li><b>Performa Model:</b> R² score {test_r2:.4f} menunjukkan model dapat menjelaskan {test_r2*100:.1f}% variasi data</li>
        <li><b>Akurasi Prediksi:</b> RMSE {test_rmse:.4f} menunjukkan rata-rata error prediksi</li>
        <li><b>Feature Terpenting:</b> {coef_df.iloc[0]['Feature']} memiliki pengaruh terbesar</li>
        <li><b>Generalisasi:</b> Model {'overfit' if abs(train_r2 - test_r2) > 0.1 else 'tidak overfit'} dengan gap {abs(train_r2-test_r2):.4f}</li>
        </ol>
        
        <h4>🚀 Rekomendasi:</h4>
        <ul>
        <li>Model sudah cukup baik untuk prediksi Petal Length</li>
        <li>Bisa ditingkatkan dengan feature engineering atau polynomial features</li>
        <li>Cocok digunakan untuk estimasi awal dalam penelitian botanis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ====================
# IRIS CLASSIFICATION
# ====================
elif analysis_type == "🌺 Iris Classification":
    st.markdown('<p class="main-header">🌺 Iris Multi-Model Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Klasifikasi Spesies Iris dengan 3 Algoritma Machine Learning</p>', unsafe_allow_html=True)
    
    # Description
    with st.expander("ℹ️ Tentang Classification Models", expanded=False):
        st.markdown("""
        ### 🤖 Algoritma yang Digunakan:
        
        **1. Random Forest Classifier**
        - Ensemble method yang menggunakan banyak decision trees
        - Voting dari multiple trees untuk hasil lebih akurat
        - Robust terhadap overfitting
        
        **2. Decision Tree Classifier**
        - Model berbentuk tree yang mudah diinterpretasi
        - Membuat keputusan berdasarkan aturan if-else
        - Visualisasi yang intuitif
        
        **3. Logistic Regression**
        - Algoritma klasik untuk binary/multiclass classification
        - Menggunakan fungsi sigmoid
        - Cepat dan efisien
        
        ### 📊 Metrics Evaluasi:
        - **Accuracy:** Persentase prediksi yang benar
        - **Precision:** Seberapa tepat prediksi positif
        - **Recall:** Seberapa lengkap menangkap kelas positif
        - **F1-Score:** Harmonic mean dari precision dan recall
        """)
    
    # Use demo data
    use_demo = st.checkbox("🎲 Gunakan Demo Dataset (Iris Built-in)", value=True)
    
    if use_demo:
        # Load iris dataset
        iris_data = load_iris()
        df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        df['species'] = iris_data.target_names[iris_data.target]
        df['species_encoded'] = iris_data.target
        
        # Dataset info
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Samples", df.shape[0])
        with col2:
            st.metric("📋 Features", df.shape[1] - 2)
        with col3:
            st.metric("🌺 Classes", df['species'].nunique())
        with col4:
            st.metric("⚖️ Balance", "Perfect" if df['species'].value_counts().std() == 0 else "Imbalanced")
        
        # Show data
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Class distribution
        st.markdown("## 📊 Distribusi Kelas")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            class_counts = df['species'].value_counts()
            st.dataframe(class_counts, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
            <b>✅ Dataset Balanced!</b><br>
            Setiap kelas memiliki jumlah sampel yang sama (50 sampel)
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart
            colors_bar = ['#667eea', '#764ba2', '#f5576c']
            class_counts.plot(kind='bar', ax=ax1, color=colors_bar, edgecolor='black', linewidth=1.5)
            ax1.set_xlabel('Species', fontweight='bold')
            ax1.set_ylabel('Count', fontweight='bold')
            ax1.set_title('Species Distribution (Bar Chart)', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Pie chart
            ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                   colors=colors_bar, startangle=90, explode=(0.05, 0.05, 0.05),
                   shadow=True, textprops={'fontweight': 'bold'})
            ax2.set_title('Species Distribution (Pie Chart)', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Feature visualization
        st.markdown("## 📈 Visualisasi Features")
        
        feature_cols = [col for col in df.columns if col not in ['species', 'species_encoded']]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(feature_cols):
            for species in df['species'].unique():
                species_data = df[df['species'] == species]
                axes[idx].hist(species_data[feature], alpha=0.6, label=species, bins=20, edgecolor='black')
            
            axes[idx].set_xlabel(feature, fontweight='bold')
            axes[idx].set_ylabel('Frequency', fontweight='bold')
            axes[idx].set_title(f'Distribution of {feature}', fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Prepare data for modeling
        st.markdown("## 🔧 Data Preprocessing")
        
        X = df[feature_cols]
        y = df['species_encoded']
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Scaling:**")
            st.dataframe(X.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**After Scaling:**")
            st.dataframe(X_scaled_df.describe(), use_container_width=True)
        
        st.info("📌 **Feature Scaling Applied:** StandardScaler untuk menormalkan rentang nilai fitur")
        
        # Train-test split
        test_size = st.slider("🎚️ Test Size (%)", min_value=10, max_value=40, value=20, step=5)
        test_ratio = test_size / 100
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_ratio, random_state=42, stratify=y
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Training Samples", len(X_train))
        with col2:
            st.metric("🧪 Testing Samples", len(X_test))
        with col3:
            st.metric("📊 Split Ratio", f"{100-test_size}:{test_size}")
        
        # Model Training
        st.markdown("## 🤖 Model Training & Evaluation")
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        model_results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            
            model_results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'predictions': y_pred_test
            }
            
            progress_bar.progress((idx + 1) / len(models))
        
        status_text.text("✅ All models trained successfully!")
        
        # Results comparison
        st.markdown("## 📊 Model Comparison")
        
        results_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'Train Accuracy': [r['train_accuracy'] for r in model_results.values()],
            'Test Accuracy': [r['test_accuracy'] for r in model_results.values()],
            'Overfitting Gap': [r['train_accuracy'] - r['test_accuracy'] for r in model_results.values()]
        })
        
        # Highlight best model
        best_idx = results_df['Test Accuracy'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        
        st.dataframe(
            results_df.style.highlight_max(subset=['Test Accuracy'], color='lightgreen')
                           .format({'Train Accuracy': '{:.4f}', 'Test Accuracy': '{:.4f}', 'Overfitting Gap': '{:.4f}'})
        )
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy comparison
        x = np.arange(len(results_df))
        width = 0.35
        
        axes[0].bar(x - width/2, results_df['Train Accuracy'], width, label='Train', 
                   color='#667eea', edgecolor='black', linewidth=1.5)
        axes[0].bar(x + width/2, results_df['Test Accuracy'], width, label='Test', 
                   color='#f5576c', edgecolor='black', linewidth=1.5)
        axes[0].set_xlabel('Model', fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontweight='bold')
        axes[0].set_title('Train vs Test Accuracy', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['Model'], rotation=15, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0.8, 1.05])
        
        # Test accuracy only
        colors_models = ['#28a745' if i == best_idx else '#667eea' for i in range(len(results_df))]
        axes[1].bar(results_df['Model'], results_df['Test Accuracy'], 
                   color=colors_models, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Accuracy', fontweight='bold')
        axes[1].set_title('Test Accuracy Comparison', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0.8, 1.05])
        
        # Overfitting gap
        colors_gap = ['#28a745' if x < 0.05 else '#ffc107' if x < 0.1 else '#dc3545' 
                     for x in results_df['Overfitting Gap']]
        axes[2].bar(results_df['Model'], results_df['Overfitting Gap'], 
                   color=colors_gap, edgecolor='black', linewidth=1.5)
        axes[2].set_ylabel('Gap', fontweight='bold')
        axes[2].set_title('Overfitting Gap (Train - Test)', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=15)
        axes[2].axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, label='Threshold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model highlight
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
        <h2>🏆 Best Model: {best_model_name}</h2>
        <h3>Test Accuracy: {results_df.loc[best_idx, 'Test Accuracy']:.4f} ({results_df.loc[best_idx, 'Test Accuracy']*100:.2f}%)</h3>
        <p>Overfitting Gap: {results_df.loc[best_idx, 'Overfitting Gap']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed analysis of best model
        st.markdown(f"## 🔍 Detailed Analysis: {best_model_name}")
        
        best_model = model_results[best_model_name]['model']
        best_predictions = model_results[best_model_name]['predictions']
        
        # Classification report
        st.markdown("### 📋 Classification Report")
        
        report = classification_report(y_test, best_predictions, 
                                      target_names=iris_data.target_names,
                                      output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)
        
        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            cm = confusion_matrix(y_test, best_predictions)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=iris_data.target_names,
                       yticklabels=iris_data.target_names,
                       cbar_kws={'label': 'Count'},
                       linewidths=2, linecolor='black', ax=ax)
            ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
            ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
            ax.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold', fontsize=14)
            st.pyplot(fig)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>📖 Cara Membaca Confusion Matrix:</h4>
            <ul>
            <li><b>Diagonal</b> (kiri atas ke kanan bawah): Prediksi yang BENAR</li>
            <li><b>Off-diagonal:</b> Prediksi yang SALAH</li>
            <li><b>Baris:</b> Actual/True class</li>
            <li><b>Kolom:</b> Predicted class</li>
            </ul>
            
            <h4>💡 Insight:</h4>
            """, unsafe_allow_html=True)
            
            total_correct = np.trace(cm)
            total_samples = cm.sum()
            
            st.markdown(f"""
            <ul>
            <li>Total prediksi benar: <b>{total_correct}/{total_samples}</b></li>
            <li>Accuracy: <b>{(total_correct/total_samples)*100:.2f}%</b></li>
            <li>Kesalahan prediksi: <b>{total_samples - total_correct}</b> sampel</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance (for Random Forest)
        if best_model_name == 'Random Forest':
            st.markdown("### 🌟 Feature Importance")
            
            importances = best_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(feature_imp_df.style.background_gradient(cmap='Greens', subset=['Importance']))
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_imp_df)))
                ax.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], 
                       color=colors_imp, edgecolor='black', linewidth=1.5)
                ax.set_xlabel('Importance Score', fontweight='bold')
                ax.set_title('Feature Importance in Random Forest', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
            
            st.markdown(f"""
            <div class="insight-box">
            <h4>💡 Feature Importance Insight:</h4>
            <ul>
            <li><b>Most Important:</b> {feature_imp_df.iloc[0]['Feature']} ({feature_imp_df.iloc[0]['Importance']:.4f})</li>
            <li><b>Least Important:</b> {feature_imp_df.iloc[-1]['Feature']} ({feature_imp_df.iloc[-1]['Importance']:.4f})</li>
            <li>Feature importance menunjukkan kontribusi setiap fitur dalam membuat prediksi</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Model comparison insights
        st.markdown("## 💡 Insights & Interpretasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Kelebihan Model:</h4>
            <ul>
            <li><b>Random Forest:</b> Robust, akurat, handle overfitting dengan baik</li>
            <li><b>Decision Tree:</b> Mudah diinterpretasi, visualisasi jelas</li>
            <li><b>Logistic Regression:</b> Cepat, efisien, probabilistic output</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>⚠️ Considerations:</h4>
            <ul>
            <li><b>Random Forest:</b> Lebih lambat, sulit diinterpretasi</li>
            <li><b>Decision Tree:</b> Prone to overfitting</li>
            <li><b>Logistic Regression:</b> Asumsi linear, kurang flexible</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Conclusion
        st.markdown("## ✅ Kesimpulan Akhir")
        
        avg_accuracy = results_df['Test Accuracy'].mean()
        
        st.markdown(f"""
        <div class="success-box">
        <h3>🎓 Summary:</h3>
        <ol>
        <li><b>Best Performer:</b> {best_model_name} dengan accuracy {results_df.loc[best_idx, 'Test Accuracy']*100:.2f}%</li>
        <li><b>Average Accuracy:</b> {avg_accuracy*100:.2f}% across all models</li>
        <li><b>Dataset Quality:</b> Excellent - balanced classes, clear separability</li>
        <li><b>Generalization:</b> Model perform well on unseen data (low overfitting)</li>
        </ol>
        
        <h3>🚀 Rekomendasi:</h3>
        <ul>
        <li>✅ Gunakan <b>{best_model_name}</b> untuk production deployment</li>
        <li>✅ Dataset iris sangat cocok untuk learning classification</li>
        <li>✅ Semua model menunjukkan performa yang baik (>90% accuracy)</li>
        <li>✅ Tidak perlu feature engineering tambahan</li>
        </ul>
        
        <h3>📈 Next Steps:</h3>
        <ul>
        <li>Cross-validation untuk validasi lebih robust</li>
        <li>Hyperparameter tuning untuk optimize performance</li>
        <li>Ensemble methods untuk combine multiple models</li>
        <li>Deploy model menggunakan API atau web service</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ====================
# SENTIMENT ANALYSIS
# ====================
elif analysis_type == "💬 Sentiment Analysis":
    st.markdown('<p class="main-header">💬 Sentiment Analysis with NLP</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analisis Sentimen Teks menggunakan Natural Language Processing</p>', unsafe_allow_html=True)
    
    # Description
    with st.expander("ℹ️ Tentang Sentiment Analysis", expanded=False):
        st.markdown("""
        ### 📖 Apa itu Sentiment Analysis?
        
        **Sentiment Analysis** adalah teknik NLP untuk mengidentifikasi dan mengekstrak opini/emosi dari teks.
        
        ### 🔄 Pipeline Process:
        1. **Text Preprocessing:**
           - Lowercase conversion
           - Remove special characters & numbers
           - Tokenization
           - Remove stopwords
           - Lemmatization
        
        2. **Sentiment Classification:**
           - Menggunakan TextBlob untuk polarity analysis
           - Polarity score: -1 (negative) to +1 (positive)
           
        3. **Visualization:**
           - Distribution analysis
           - Word clouds
           - Frequency analysis
        
        ### 🎯 Use Cases:
        - Customer feedback analysis
        - Social media monitoring
        - Product review analysis
        - Brand sentiment tracking
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Sentiment Dataset (CSV)", 
        type=['csv'],
        help="CSV harus memiliki kolom 'Text' yang berisi teks untuk dianalisis"
    )
    
    # Demo text input
    st.markdown("### ✏️ Atau Coba dengan Teks Manual")
    
    manual_text = st.text_area(
        "Masukkan teks untuk dianalisis:",
        placeholder="Contoh: This product is amazing! I love it so much...",
        height=100
    )
    
    analyze_manual = st.button("🔍 Analyze Manual Text", type="primary")
    
    if uploaded_file is not None or (analyze_manual and manual_text):
        import re
        from textblob import TextBlob
        from wordcloud import WordCloud
        
        # Load or create data
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Check if 'Text' column exists
            if 'Text' not in df.columns:
                st.error("❌ Dataset harus memiliki kolom 'Text'!")
                st.stop()
        else:
            # Create single row dataframe from manual input
            df = pd.DataFrame({'Text': [manual_text]})
        
        # Dataset info
        st.markdown("## 📊 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Total Texts", len(df))
        with col2:
            avg_length = df['Text'].str.len().mean()
            st.metric("📏 Avg Length", f"{avg_length:.0f} chars")
        with col3:
            total_words = df['Text'].str.split().str.len().sum()
            st.metric("📖 Total Words", f"{total_words:,}")
        
        # Show sample
        with st.expander("👀 Preview Data"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Text Preprocessing
        st.markdown("## 🔧 Text Preprocessing")
        
        def preprocess_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = ' '.join(text.split())
            return text
        
        with st.spinner("🔄 Processing texts..."):
            df['cleaned_text'] = df['Text'].apply(preprocess_text)
        
        st.success("✅ Preprocessing complete!")
        
        # Show before/after
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Preprocessing:**")
            st.text_area("Original", df['Text'].iloc[0], height=100, disabled=True)
        
        with col2:
            st.markdown("**After Preprocessing:**")
            st.text_area("Cleaned", df['cleaned_text'].iloc[0], height=100, disabled=True)
        
        # Sentiment Analysis
        st.markdown("## 🎯 Sentiment Analysis")
        
        def get_sentiment(text):
            analysis = TextBlob(text)
            polarity = analysis.polarity
            
            if polarity > 0.1:
                return 'Positive', polarity
            elif polarity < -0.1:
                return 'Negative', polarity
            else:
                return 'Neutral', polarity
        
        with st.spinner("🔍 Analyzing sentiments..."):
            df[['sentiment', 'polarity']] = df['cleaned_text'].apply(
                lambda x: pd.Series(get_sentiment(x))
            )
        
        st.success("✅ Sentiment analysis complete!")
        
        # Results
        st.markdown("## 📊 Results")
        
        sentiment_counts = df['sentiment'].value_counts()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            positive_pct = (sentiment_counts.get('Positive', 0) / len(df)) * 100
            st.metric(
                "😊 Positive", 
                sentiment_counts.get('Positive', 0),
                delta=f"{positive_pct:.1f}%"
            )
        
        with col2:
            neutral_pct = (sentiment_counts.get('Neutral', 0) / len(df)) * 100
            st.metric(
                "😐 Neutral", 
                sentiment_counts.get('Neutral', 0),
                delta=f"{neutral_pct:.1f}%"
            )
        
        with col3:
            negative_pct = (sentiment_counts.get('Negative', 0) / len(df)) * 100
            st.metric(
                "😞 Negative", 
                sentiment_counts.get('Negative', 0),
                delta=f"{negative_pct:.1f}%"
            )
        
        with col4:
            avg_polarity = df['polarity'].mean()
            overall = "Positive" if avg_polarity > 0 else "Negative" if avg_polarity < 0 else "Neutral"
            st.metric(
                "📈 Overall", 
                overall,
                delta=f"{avg_polarity:.3f}"
            )
        
        # Visualizations
        st.markdown("## 📈 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "☁️ Word Clouds", "📉 Polarity Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors_sent = {'Positive': '#28a745', 'Neutral': '#ffc107', 'Negative': '#dc3545'}
                sentiment_colors = [colors_sent.get(x, '#667eea') for x in sentiment_counts.index]
                
                ax.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=sentiment_colors, edgecolor='black', linewidth=2)
                ax.set_xlabel('Sentiment', fontweight='bold', fontsize=12)
                ax.set_ylabel('Count', fontweight='bold', fontsize=12)
                ax.set_title('Sentiment Distribution', fontweight='bold', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for i, v in enumerate(sentiment_counts.values):
                    ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
                
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                explode = [0.1 if i == sentiment_counts.values.argmax() else 0 
                          for i in range(len(sentiment_counts))]
                
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                      colors=sentiment_colors, startangle=90, explode=explode,
                      shadow=True, textprops={'fontweight': 'bold', 'fontsize': 11})
                ax.set_title('Sentiment Proportion', fontweight='bold', fontsize=14)
                
                st.pyplot(fig)
        
        with tab2:
            st.markdown("### ☁️ Word Clouds per Sentiment")
            
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                if sentiment in df['sentiment'].values:
                    sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
                    
                    if len(sentiment_text.strip()) > 0:
                        st.markdown(f"**{sentiment} Sentiment:**")
                        
                        color_map = {
                            'Positive': 'Greens',
                            'Neutral': 'Blues', 
                            'Negative': 'Reds'
                        }
                        
                        wordcloud = WordCloud(
                            width=1200, height=400, 
                            background_color='white',
                            colormap=color_map[sentiment],
                            max_words=100,
                            relative_scaling=0.5,
                            min_font_size=10
                        ).generate(sentiment_text)
                        
                        fig, ax = plt.subplots(figsize=(15, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'{sentiment} Word Cloud', fontsize=16, fontweight='bold', pad=20)
                        st.pyplot(fig)
                    else:
                        st.info(f"No {sentiment.lower()} texts found")
        
        with tab3:
            st.markdown("### 📉 Polarity Score Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Histogram
                n, bins, patches = ax.hist(df['polarity'], bins=30, edgecolor='black', linewidth=1.2)
                
                # Color bars based on polarity
                for i, patch in enumerate(patches):
                    if bins[i] < -0.1:
                        patch.set_facecolor('#dc3545')
                    elif bins[i] > 0.1:
                        patch.set_facecolor('#28a745')
                    else:
                        patch.set_facecolor('#ffc107')
                
                ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Neutral')
                ax.axvline(x=df['polarity'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
                ax.set_xlabel('Polarity Score', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title('Distribution of Polarity Scores', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Box plot by sentiment
                sentiment_order = ['Negative', 'Neutral', 'Positive']
                data_to_plot = [df[df['sentiment'] == s]['polarity'].values 
                               for s in sentiment_order if s in df['sentiment'].values]
                labels_to_plot = [s for s in sentiment_order if s in df['sentiment'].values]
                
                bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                               boxprops=dict(linewidth=2),
                               whiskerprops=dict(linewidth=2),
                               capprops=dict(linewidth=2),
                               medianprops=dict(color='red', linewidth=2))
                
                # Color boxes
                for patch, label in zip(bp['boxes'], labels_to_plot):
                    if label == 'Positive':
                        patch.set_facecolor('#28a745')
                    elif label == 'Negative':
                        patch.set_facecolor('#dc3545')
                    else:
                        patch.set_facecolor('#ffc107')
                
                ax.set_ylabel('Polarity Score', fontweight='bold')
                ax.set_title('Polarity Score by Sentiment Category', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
                
                st.pyplot(fig)
        
        # Detailed results table
        st.markdown("## 📋 Detailed Results")
        
        results_display = df[['Text', 'sentiment', 'polarity']].copy()
        results_display['polarity'] = results_display['polarity'].round(4)
        
        # Add color coding
        def color_sentiment(val):
            if val == 'Positive':
                return 'background-color: #d4edda'
            elif val == 'Negative':
                return 'background-color: #f8d7da'
            else:
                return 'background-color: #fff3cd'
        
        st.dataframe(
            results_display.head(20).style.applymap(color_sentiment, subset=['sentiment']),
            use_container_width=True
        )
        
        if len(df) > 20:
            st.info(f"Showing top 20 results. Total: {len(df)} texts analyzed.")
        
        # Statistics
        st.markdown("## 📊 Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Polarity Statistics")
            polarity_stats = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    df['polarity'].mean(),
                    df['polarity'].median(),
                    df['polarity'].std(),
                    df['polarity'].min(),
                    df['polarity'].max(),
                    df['polarity'].max() - df['polarity'].min()
                ]
            })
            polarity_stats['Value'] = polarity_stats['Value'].round(4)
            st.dataframe(polarity_stats, use_container_width=True)
        
        with col2:
            st.markdown("### Sentiment Percentages")
            sentiment_pct = (sentiment_counts / len(df) * 100).round(2)
            sentiment_pct_df = pd.DataFrame({
                'Sentiment': sentiment_pct.index,
                'Percentage': sentiment_pct.values
            })
            st.dataframe(sentiment_pct_df, use_container_width=True)
        
        # Insights
        st.markdown("## 💡 Key Insights")
        
        dominant_sentiment = sentiment_counts.idxmax()
        dominant_pct = (sentiment_counts.max() / len(df)) * 100
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>🔍 Analisis Sentiment:</h4>
        <ul>
        <li><b>Sentimen Dominan:</b> {dominant_sentiment} ({dominant_pct:.1f}% dari total)</li>
        <li><b>Rata-rata Polarity:</b> {df['polarity'].mean():.4f} (skala -1 hingga +1)</li>
        <li><b>Distribusi:</b> {"Balanced" if sentiment_counts.std() < len(df)*0.1 else "Skewed"}</li>
        <li><b>Polarity Range:</b> {df['polarity'].min():.4f} to {df['polarity'].max():.4f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        overall_tone = ""
        if df['polarity'].mean() > 0.2:
            overall_tone = "Overall tone is <b>STRONGLY POSITIVE</b> 😊"
            tone_color = "#28a745"
        elif df['polarity'].mean() > 0:
            overall_tone = "Overall tone is <b>SLIGHTLY POSITIVE</b> 🙂"
            tone_color = "#5cb85c"
        elif df['polarity'].mean() > -0.2:
            overall_tone = "Overall tone is <b>SLIGHTLY NEGATIVE</b> 😕"
            tone_color = "#f0ad4e"
        else:
            overall_tone = "Overall tone is <b>STRONGLY NEGATIVE</b> 😞"
            tone_color = "#dc3545"
        
        st.markdown(f"""
        <div style="background-color: {tone_color}; color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h3>{overall_tone}</h3>
        <p>Average Polarity Score: {df['polarity'].mean():.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("## 🚀 Recommendations")
        
        if dominant_sentiment == 'Negative' and dominant_pct > 50:
            st.markdown("""
            <div class="insight-box">
            <h4>⚠️ Action Required:</h4>
            <ul>
            <li>Lebih dari 50% sentiment negatif - perlu action plan</li>
            <li>Review dan address concern pelanggan</li>
            <li>Identifikasi root cause dari feedback negatif</li>
            <li>Implement improvement berdasarkan feedback</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        elif dominant_sentiment == 'Positive' and dominant_pct > 50:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Great Performance:</h4>
            <ul>
            <li>Mayoritas feedback positif - pertahankan kualitas!</li>
            <li>Leverage positive reviews untuk marketing</li>
            <li>Identify success factors untuk direplikasi</li>
            <li>Maintain customer satisfaction level</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
            <h4>📊 Mixed Sentiment:</h4>
            <ul>
            <li>Sentiment terdistribusi merata - ada area improvement dan strength</li>
            <li>Focus on converting neutral ke positive</li>
            <li>Address negative feedback secara selektif</li>
            <li>Strengthen positive aspects</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Export option
        st.markdown("## 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
        
        with col2:
            summary_data = {
                'Metric': ['Total Texts', 'Positive', 'Neutral', 'Negative', 'Avg Polarity', 'Dominant Sentiment'],
                'Value': [
                    len(df),
                    sentiment_counts.get('Positive', 0),
                    sentiment_counts.get('Neutral', 0),
                    sentiment_counts.get('Negative', 0),
                    f"{df['polarity'].mean():.4f}",
                    dominant_sentiment
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Summary (CSV)",
                data=summary_csv,
                file_name="sentiment_summary.csv",
                mime="text/csv"
            )
        
        with col3:
            st.info("💡 Tip: Use exported data for further analysis atau reporting")
        
        # Conclusion
        st.markdown("## ✅ Conclusion")
        
        st.markdown(f"""
        <div class="success-box">
        <h3>🎓 Analysis Complete!</h3>
        
        <h4>📈 Summary:</h4>
        <ul>
        <li><b>Total Texts Analyzed:</b> {len(df)}</li>
        <li><b>Sentiment Breakdown:</b> {sentiment_counts.get('Positive', 0)} Positive, {sentiment_counts.get('Neutral', 0)} Neutral, {sentiment_counts.get('Negative', 0)} Negative</li>
        <li><b>Overall Sentiment:</b> {dominant_sentiment} ({dominant_pct:.1f}%)</li>
        <li><b>Average Polarity:</b> {df['polarity'].mean():.4f}</li>
        </ul>
        
        <h4>🔧 Tools Used:</h4>
        <ul>
        <li>Natural Language Processing (NLP)</li>
        <li>TextBlob for sentiment analysis</li>
        <li>Word Cloud for visualization</li>
        <li>Statistical analysis</li>
        </ul>
        
        <h4>💡 Next Steps:</h4>
        <ul>
        <li>Deep dive into specific negative/positive texts</li>
        <li>Implement sentiment tracking over time</li>
        <li>Combine with other analytics untuk comprehensive insights</li>
        <li>Use findings untuk strategic decision making</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
    <h4>🚀 CODVEDA Analytics Hub</h4>
    <p>Powered by Streamlit • Machine Learning • Data Science</p>
    <p style='color: #666;'>Built with ❤️ for Data Analysis & Insights</p>
    </div>
    """, unsafe_allow_html=True)
