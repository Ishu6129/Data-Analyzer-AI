import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def show_plots(data, analyzer=None):
    # Set up columns with proper spacing
    col1, col2 = st.columns(2, gap="large")
    
    # Numeric Features
    numeric_cols = data.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        with col1:
            with st.container():
                st.markdown("#### Numeric Distributions")
                selected_num = st.selectbox(
                    "Select numeric column:", 
                    numeric_cols,
                    key="num_dist_select"
                )
                fig = px.histogram(
                    data, 
                    x=selected_num,
                    marginal="box",
                    color_discrete_sequence=['#6eb5ff']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.container():
                st.markdown("#### Outlier Detection")
                fig = px.box(
                    data,
                    y=numeric_cols[:min(3, len(numeric_cols))],
                    color_discrete_sequence=['#ff7f7f']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Features
    cat_cols = data.select_dtypes(exclude=np.number).columns
    if len(cat_cols) > 0:
        with st.container():
            st.markdown("#### Categorical Analysis")
            selected_cat = st.selectbox(
                "Select categorical column:", 
                cat_cols,
                key="cat_analysis_select"
            )
            fig = px.bar(
                data[selected_cat].value_counts(),
                color_discrete_sequence=['#6eb5ff'],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)