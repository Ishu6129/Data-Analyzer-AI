import google.generativeai as genai
import streamlit as st
from datetime import datetime, timedelta
import os
import json
import numpy as np
from typing import Dict, Any


def configure_gemini() -> genai.GenerativeModel:
    """Configure and return the Gemini model instance"""
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Missing Gemini API key")
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro-latest')

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

def get_ai_suggestions(data_summary: str, ml_readiness: Dict) -> str:
    """
    Generate concise, actionable AI recommendations for data science workflow
    
    Args:
        data_summary: Summary of the dataset
        ml_readiness: Dictionary containing ML readiness metrics
        
    Returns:
        Formatted markdown string with recommendations
    """
    try:
        model = configure_gemini()
        if not model:
            return "⚠️ API key not configured"
        
        # Convert numpy types for JSON serialization
        ml_readiness = convert_numpy_types(ml_readiness)
        
        prompt = f"""
        You are a senior data scientist providing concise recommendations to a business analyst.
        Dataset summary: {data_summary[:2000]}... [truncated]
        Key metrics: {json.dumps({k: ml_readiness[k] for k in sorted(ml_readiness)[:5]}, indent=2)}

        Provide ONLY the most critical 2-3 recommendations per category in this exact format:
        
        ### 🔧 Data Cleaning
        - [Most important cleaning task]
        - [Second most important]
        
        ### 🛠️ Feature Tips
        - [Top feature engineering suggestion]
        - [Second suggestion]
        
        ### 🤖 Model Advice
        - [Best algorithm choice]
        - [Alternative option]
        
        ### ⚠️ Watch Outs
        - [Main potential issue]
        - [Secondary concern]
        
        Rules:
        - Use simple language
        - No technical jargon
        - Max 15 words per bullet
        - Skip categories if no relevant suggestions
        - Never exceed 15 bullet points total
        - Prioritize actionable items
        - Focus on business impact
        """
        
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            return "🚧 No recommendations generated - try again later"
            
        # Post-process to ensure concise output
        recommendations = response.text.strip()
        if len(recommendations.split('\n')) > 15:
            recommendations = '\n'.join(recommendations.split('\n')[:15]) + "\n..."
            
        return recommendations
        
    except Exception as e:
        return f"❌ Error: {str(e)[:200]}"