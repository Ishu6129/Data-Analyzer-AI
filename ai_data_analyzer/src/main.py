import streamlit as st
from analyzer import DataAnalyzer
from utils.visualization import show_plots
from utils.model_evaluation import display_model_metrics
from utils.ai_helper import get_ai_suggestions
import pandas as pd
import plotly.express as px
import numpy as np

def configure_page():
    st.set_page_config(
        page_title="DataPulse Pro",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Exo+2:wght@400;600&display=swap');
    
    /* Cyberpunk-inspired color scheme */
    :root {
        --primary: #00f0ff;
        --secondary: #ff2d75;
        --dark: #0a0a20;
        --darker: #050510;
        --neon-glow: 0 0 10px var(--primary), 0 0 20px var(--primary);
    }
    
    /* Main app styling */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    .stApp {
        background: var(--darker) !important;
        font-family: 'Exo 2', sans-serif;
        color: white !important;
    }
    
    /* Futuristic header with animated gradient */
    .main-header {
        background: linear-gradient(89deg, #0a0a20 0%, #1e1e40 50%, #0a0a20 100%);
        color: white;
        padding: 3rem 1rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 1px solid var(--primary);
        position: relative;
        overflow: hidden;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        animation: headerGlow 8s infinite alternate;
    }
    
    @keyframes headerGlow {
        0% { box-shadow: 0 0 5px rgba(0, 240, 255, 0.3); }
        100% { box-shadow: 0 0 9px rgba(0, 240, 255, 0.7); }
    }
    
    .main-header h1 {
        font-size: 3.2rem;
        margin: 0;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.4rem;
        margin: 0.8rem 0 0;
        color: rgba(255,255,255,0.8);
        font-family: 'Exo 2', sans-serif;
    }
    
    /* Futuristic cards with glass morphism */
    .stContainer {
        background: rgba(15, 15, 35, 0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 240, 255, 0.2);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stContainer:hover {
        border: 1px solid var(--primary);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }
    
    /* Cyberpunk buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: var(--dark) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--neon-glow) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 0 20px var(--primary) !important;
    }
    
    /* Futuristic tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(0, 240, 255, 0.3) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 15, 35, 0.5) !important;
        color: white !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        border: 1px solid rgba(0, 240, 255, 0.2) !important;
        border-bottom: none !important;
        font-family: 'Exo 2', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: var(--dark) !important;
        font-weight: 600 !important;
        box-shadow: var(--neon-glow) !important;
        border: none !important;
    }
    
    /* Futuristic file uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--primary) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        background: rgba(15, 15, 35, 0.5) !important;
        backdrop-filter: blur(5px);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--darker);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    /* Animated background elements */
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 100%;
        background: radial-gradient(circle at 20% 30%, rgba(0, 240, 255, 0.1) 0%, transparent 50%);
        z-index: -1;
        animation: pulseBackground 15s infinite alternate;
    }
    
    @keyframes pulseBackground {
        0% { opacity: 0.3; }
        100% { opacity: 0.7; }
    }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>🚀 DATAPULSE PRO</h1>
        <p>AI-Powered Data Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

def analyze_data(uploaded_file):
    try:
        with st.spinner("🔍 Analyzing your dataset..."):
            analyzer = DataAnalyzer(uploaded_file)
            analyzer.analyze()
            return analyzer
    except Exception as e:
        st.error(f"⚠️ Error processing file: {str(e)}")
        return None

def display_ml_readiness(analyzer):
    with st.container():
        st.markdown("### 🏆 Data Quality Report")
        
        tab1, tab2 = st.tabs(["🔍 Missing Values & Outliers", "📈 Feature Insights"])
        
        with tab1:
            # ===== Modern Missing Values Analysis =====
            st.markdown("#### 🧩 Missing Values Analysis")
            
            # Calculate missing data with explicit type conversion
            missing_data = analyzer.data.isnull().sum()
            missing_percent = (missing_data / len(analyzer.data)) * 100
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values.astype(int),
                'Percentage (%)': missing_percent.values.astype(float).round(1),
                'Data Type': analyzer.data.dtypes.astype(str).values
            }).sort_values('Percentage (%)', ascending=False)
            
            # Filter only columns with missing values
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if not missing_df.empty:
                # ---- Summary Cards ----
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Affected Columns", 
                            f"{len(missing_df)}/{len(analyzer.data.columns)}",
                            help="Columns with missing values")
                
                with cols[1]:
                    avg_missing = float(missing_df['Percentage (%)'].mean())
                    st.metric("Avg. Missing", 
                            f"{avg_missing:.1f}%",
                            delta_color="inverse",
                            delta=f"{(avg_missing - float(missing_df['Percentage (%)'].median())):.1f}% from median")
                
                with cols[2]:
                    st.metric("Total Missing", 
                            f"{int(missing_df['Missing Values'].sum()):,} cells",
                            help="Total empty cells in dataset")
                
                # ---- Interactive Visualization ----
                fig = px.bar(
                    missing_df,
                    x='Percentage (%)',
                    y='Column',
                    color='Percentage (%)',
                    color_continuous_scale='sunset',
                    orientation='h',
                    height=max(400, 35 * len(missing_df)),
                    labels={'Percentage (%)': 'Missing Values (%)'},
                    hover_data=['Data Type']
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis={'categoryorder':'total ascending'},
                    xaxis_range=[0, 100],
                    hoverlabel=dict(bgcolor="white")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ---- Smart Recommendations ----
                with st.expander("💡 Treatment Recommendations", expanded=True):
                    if avg_missing > 50:
                        st.warning("**Critical Issue** - High missing data detected:")
                        st.write("""
                        - 🗑️ **Drop columns** with >70% missing values  
                        - 🔍 **Investigate data collection** for systematic issues  
                        - 🏷️ Consider **flagging missingness** as a feature  
                        """)
                    elif avg_missing > 20:
                        st.warning("**Moderate Issue** - Significant missing data:")
                        st.write("""
                        - 🧠 **Advanced imputation**: KNN or MICE for >20% missing  
                        - 📌 **Create indicators** for missing patterns  
                        - 📊 **Analyze patterns**: Is missingness random?  
                        """)
                    else:
                        st.success("**Manageable Missing Data** - Suggested actions:")
                        st.write("""
                        - 🧮 **Numerical features**: Impute with median/mean  
                        - 🔠 **Categorical features**: Use mode or 'Missing' category  
                        - ✅ **Verify impact** after imputation  
                        """)
                
                # ---- Data Table ----
                with st.expander("📋 Detailed Missing Data Table"):
                    st.dataframe(
                        missing_df.style.format({'Percentage (%)': '{:.1f}%'})
                        .background_gradient(subset=['Percentage (%)'], cmap='YlOrRd')
                        .set_properties(**{'background-color': 'black', 'color': 'white'}),
                        height=300,
                        use_container_width=True
                    )
            else:
                st.success("✅ **Perfect Data!** No missing values found.", icon="🎉")
                st.balloons()
            
            # ===== Enhanced Outlier Detection =====
            st.markdown("#### 📊 Outlier Detection")
            outliers = analyzer.ml_readiness['outlier_analysis']
            
            if isinstance(outliers, dict) and outliers:
                # Convert all values to native Python types
                outlier_data = []
                for col, values in outliers.items():
                    outlier_data.append({
                        'Column': col,
                        'Outlier Count': int(values['outlier_count']),
                        'Outlier %': float(values['outlier_pct'])
                    })
                outlier_df = pd.DataFrame(outlier_data)
                
                # Summary cards
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Features with Outliers", 
                            f"{len(outlier_df)}/{len(analyzer.data.columns)}")
                
                with cols[1]:
                    avg_outliers = float(outlier_df['Outlier %'].mean())
                    st.metric("Avg. Outliers", 
                            f"{avg_outliers:.1f}%",
                            delta_color="inverse")
                
                with cols[2]:
                    st.metric("Total Outliers", 
                            f"{int(outlier_df['Outlier Count'].sum()):,} points")
                
                # Interactive visualization
                fig = px.bar(
                    outlier_df.sort_values('Outlier %', ascending=False),
                    x='Outlier %',
                    y='Column',
                    color='Outlier %',
                    color_continuous_scale='thermal',
                    orientation='h',
                    labels={'Outlier %': 'Outliers (%)'},
                    height=max(400, 35 * len(outlier_df)))
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                with st.expander("🔧 Outlier Treatment Suggestions"):
                    if avg_outliers > 5:
                        st.warning("**Significant Outliers** detected:")
                        st.write("""
                        - 📏 **Winsorize** extreme values (cap at 95th percentile)  
                        - 🔢 **Transformations**: Log, square root for skewed data  
                        - 🧐 **Investigate**: Are these valid anomalies?  
                        """)
                    else:
                        st.success("**Minor Outliers** - Typical treatment options:")
                        st.write("""
                        - 🏷️ **Keep as-is** if they represent valid cases  
                        - ✂️ **Trim** only extreme outliers  
                        - 🔄 **Robust models**: Random Forests, SVM  
                        """)
                
                # Data table
                with st.expander("📑 Outlier Details"):
                    st.dataframe(
                        outlier_df.style.format({'Outlier %': '{:.1f}%'})
                        .background_gradient(cmap='YlOrRd'),
                        use_container_width=True
                    )
            else:
                st.info("🔍 No significant outliers detected in numeric features")

        with tab2:
            # ===== Feature Insights =====
            st.markdown("#### 📈 Feature Importance")
            numeric_cols = analyzer.data.select_dtypes(include=np.number).columns
            
            if len(numeric_cols) > 1:
                try:
                    if 'feature_importance' not in analyzer.ml_readiness or not analyzer.ml_readiness['feature_importance']:
                        corr_matrix = analyzer.data[numeric_cols].corr().abs()
                        feature_imp = corr_matrix.mean().sort_values(ascending=False).astype(float)
                    else:
                        feature_imp = pd.Series({
                            k: float(v) for k, v in analyzer.ml_readiness['feature_importance'].items()
                        }).sort_values(ascending=False)
                    
                    # Visualize top 20 features
                    fig = px.bar(
                        feature_imp.head(20),
                        orientation='h',
                        color=feature_imp.head(20).values.astype(float),
                        color_continuous_scale='deep',
                        labels={'index':'Feature','value':'Importance'},
                        title='Top 20 Important Features (Mean Absolute Correlation)',
                        height=500
                    )
                    fig.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature correlations
                    st.markdown("#### 🔗 Feature Correlations")
                    corr_matrix = analyzer.data[numeric_cols].corr().astype(float)
                    
                    fig = px.imshow(
                        corr_matrix,
                        color_continuous_scale='RdBu',
                        zmin=-1,
                        zmax=1,
                        height=600,
                        title='Feature Correlation Matrix'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation insights
                    with st.expander("💎 Correlation Insights"):
                        high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix.abs() < 1.0)
                        if high_corr.any().any():
                            st.warning("**High Correlations** detected (|r| > 0.8):")
                            st.write("""
                            - 🧹 Consider **removing redundant features**  
                            - 🧩 **Feature engineering**: Create composite features  
                            - 📉 **Regularization**: Helps with multicollinearity  
                            """)
                        else:
                            st.success("**Good Feature Independence**: No extremely high correlations")
                
                except Exception as e:
                    st.error(f"Feature analysis error: {str(e)}")
            else:
                st.warning("Need at least 2 numeric features for importance analysis")

def main():
    configure_page()
    show_header()
    
    uploaded_file = st.file_uploader(
        "📤 Upload your dataset (CSV, Excel, JSON)",
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, Excel, JSON files"
    )
    
    if uploaded_file:
        analyzer = analyze_data(uploaded_file)
        
        if analyzer:
            col1, col2 = st.columns([7, 3], gap="large")
            
            with col1:
                with st.container():
                    st.markdown("### 📋 Data Preview")
                    st.dataframe(analyzer.data.head(), use_container_width=True)
                
                with st.container():
                    st.markdown("### 📊 Data Visualizations")
                    show_plots(analyzer.data, analyzer)
                
                with st.container():
                    st.markdown("### 🧠 Model Evaluation")
                    display_model_metrics(analyzer.data)
            
            with col2:
                with st.container():
                    display_ml_readiness(analyzer)
                
                with st.container():
                    st.markdown("### 💡 AI Recommendations")
                    if st.button("Get Smart Suggestions", type="primary"):
                        with st.spinner("🧠 Generating insights..."):
                            suggestions = get_ai_suggestions(analyzer.summary, analyzer.ml_readiness)
                            with st.expander("View Recommendations", expanded=True):
                                st.markdown(suggestions)

if __name__ == "__main__":
    main()