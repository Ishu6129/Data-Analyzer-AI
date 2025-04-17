import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, mean_squared_error, r2_score, mean_absolute_error
)

MODEL_OPTIONS = {
    "Classification": {
        "Random Forest": "rf_classifier",
        "XGBoost": "xgb_classifier",
        "Logistic Regression": "logistic",
        "Support Vector Machine": "svm",
        "Decision Tree": "decision_tree"
    },
    "Regression": {
        "Random Forest": "rf_regressor",
        "XGBoost": "xgb_regressor",
        "Linear Regression": "linear",
        "Support Vector Regression": "svr",
        "Decision Tree": "decision_tree"
    }
}

def get_download_link(file_content, filename):
    """Generate a download link for the file"""
    b64 = base64.b64encode(file_content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'

def generate_model_code(model_name, problem_type, X_train, y_train, X_test, y_test, metrics, target_col):
    """Generate complete executable Python code for the model"""
    model_var_name = model_name.lower().replace(" ", "_")
    
    # Model-specific implementation code
    if problem_type == "Classification":
        if model_name == "Random Forest":
            model_code = f"""
from sklearn.ensemble import RandomForestClassifier
{model_var_name} = RandomForestClassifier(random_state=42)
"""
        elif model_name == "XGBoost":
            model_code = f"""
from xgboost import XGBClassifier
{model_var_name} = XGBClassifier(eval_metric='mlogloss', random_state=42)
"""
        elif model_name == "Logistic Regression":
            model_code = f"""
from sklearn.linear_model import LogisticRegression
{model_var_name} = LogisticRegression(max_iter=1000, random_state=42)
"""
        elif model_name == "Support Vector Machine":
            model_code = f"""
from sklearn.svm import SVC
{model_var_name} = SVC(probability=True, random_state=42)
"""
        elif model_name == "Decision Tree":
            model_code = f"""
from sklearn.tree import DecisionTreeClassifier
{model_var_name} = DecisionTreeClassifier(random_state=42)
"""
    else:  # Regression
        if model_name == "Random Forest":
            model_code = f"""
from sklearn.ensemble import RandomForestRegressor
{model_var_name} = RandomForestRegressor(random_state=42)
"""
        elif model_name == "XGBoost":
            model_code = f"""
from xgboost import XGBRegressor
{model_var_name} = XGBRegressor(random_state=42)
"""
        elif model_name == "Linear Regression":
            model_code = f"""
from sklearn.linear_model import LinearRegression
{model_var_name} = LinearRegression()
"""
        elif model_name == "Support Vector Regression":
            model_code = f"""
from sklearn.svm import SVR
{model_var_name} = SVR()
"""
        elif model_name == "Decision Tree":
            model_code = f"""
from sklearn.tree import DecisionTreeRegressor
{model_var_name} = DecisionTreeRegressor(random_state=42)
"""

    # Dataset loading code
    load_code = f"""
# Load dataset (user should replace this with their dataset loading code)
import pandas as pd
#---------------------------------------------------------------------------------------------
data = pd.read_csv('your_dataset.csv')  # Example - user should modify this
#---------------------------------------------------------------------------------------------
X = data.drop(columns=['{target_col}'])
y = data['{target_col}']
"""

    # Feature engineering code
    feature_code = f"""
# Feature Engineering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Create preprocessing pipelines
numeric_transformer = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
"""

    # Training and evaluation code
    train_code = f"""
# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create complete pipeline
pipeline = make_pipeline(preprocessor, {model_var_name})

# Train the model
print("\\n=== Training {model_name} ===")
pipeline.fit(X_train, y_train)
"""

    # Full generated code
    full_code = f"""# Auto-generated Model Code for {model_name} ({problem_type})
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *

{load_code}

{feature_code}

{model_code.strip()}

{train_code}

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluation Metrics
print("\\n=== Model Evaluation ===")
{generate_metrics_code(problem_type, model_var_name)}

# Visualization
{generate_visualization_code(problem_type, model_var_name)}

# Save model
import joblib
joblib.dump(pipeline, '{model_var_name}_model.pkl')
print("\\nModel saved as '{model_var_name}_model.pkl'")
"""
    return full_code

def generate_metrics_code(problem_type, model_var_name):
    """Generate metrics evaluation code"""
    if problem_type == "Classification":
        return f"""print(f"Accuracy: {{accuracy_score(y_test, y_pred):.2%}}")
print(f"Precision: {{precision_score(y_test, y_pred, average='weighted'):.2%}}")
print(f"Recall: {{recall_score(y_test, y_pred, average='weighted'):.2%}}")
print(f"F1 Score: {{f1_score(y_test, y_pred, average='weighted'):.2%}}")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()"""
    else:
        return f"""print(f"R² Score: {{r2_score(y_test, y_pred):.3f}}")
print(f"RMSE: {{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}}")
print(f"MAE: {{mean_absolute_error(y_test, y_pred):.3f}}")

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()"""

def generate_visualization_code(problem_type, model_var_name):
    """Generate visualization code"""
    if problem_type == "Classification":
        return f"""# Classification Report
from sklearn.metrics import classification_report
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve (for binary classification)
if len(np.unique(y_test)) == 2:
    from sklearn.metrics import roc_curve, auc
    y_prob = {model_var_name}.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {{roc_auc:.2f}})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()"""
    else:
        return """# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()"""

def download_models(selected_models, problem_type, X_train, y_train, X_test, y_test, metrics_dict, target_col):
    """Create download buttons for all selected models with complete code"""
    with st.expander("💾 Download Complete Model Code", expanded=True):
        st.write("Download ready-to-run Python scripts for each model:")
        
        for model_name in selected_models:
            if model_name in metrics_dict:
                code = generate_model_code(
                    model_name, problem_type, 
                    X_train, y_train, X_test, y_test,
                    metrics_dict[model_name],
                    target_col
                )
                filename = f"{model_name.replace(' ', '_')}_complete.py"
                st.markdown(get_download_link(code, filename), unsafe_allow_html=True)

def get_model(model_name, problem_type):
    """Return model instance based on selection"""
    if problem_type == "Classification":
        if model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier()
        elif model_name == "XGBoost":
            from xgboost import XGBClassifier
            return XGBClassifier(eval_metric='mlogloss')
        elif model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000)
        elif model_name == "Support Vector Machine":
            from sklearn.svm import SVC
            return SVC(probability=True)
        elif model_name == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier()
    else:
        if model_name == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor()
        elif model_name == "XGBoost":
            from xgboost import XGBRegressor
            return XGBRegressor()
        elif model_name == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        elif model_name == "Support Vector Regression":
            from sklearn.svm import SVR
            return SVR()
        elif model_name == "Decision Tree":
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor()

def display_predictions(model, X_test, y_test, problem_type):
    """Display model predictions in a user-friendly format"""
    with st.expander("🔮 Model Predictions", expanded=True):
        y_pred = model.predict(X_test)
        
        if problem_type == "Classification":
            results = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred,
                "Correct": y_test == y_pred
            })
            color_col = "Correct"
        else:
            results = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred,
                "Difference": np.abs(y_test - y_pred)
            })
            color_col = "Difference"
        
        st.dataframe(results.head(10).style.format("{:.3f}"))
        
        fig = px.scatter(
            results.head(50),
            x="Actual",
            y="Predicted",
            color=color_col,
            color_continuous_scale=px.colors.diverging.RdYlGn[::-1],
            title="Actual vs Predicted Values"
        )
        fig.add_shape(type="line", 
                     x0=results["Actual"].min(), 
                     y0=results["Actual"].min(),
                     x1=results["Actual"].max(), 
                     y1=results["Actual"].max())
        st.plotly_chart(fig, use_container_width=True)

def evaluate_models(X, y_encoded, problem_type, selected_models):
    """Evaluate selected models"""
    numeric_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    results = []
    for model_name in selected_models:
        try:
            model = get_model(model_name, problem_type)
            pipeline = make_pipeline(preprocessor, model)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            with st.spinner(f"Training {model_name}..."):
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
            
            if problem_type == "Classification":
                metrics = {
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted'),
                    "Recall": recall_score(y_test, y_pred, average='weighted'),
                    "F1 Score": f1_score(y_test, y_pred, average='weighted')
                }
            else:
                metrics = {
                    "Model": model_name,
                    "R² Score": r2_score(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAE": mean_absolute_error(y_test, y_pred)
                }
            
            results.append(metrics)
            
            if st.checkbox(f"Show predictions for {model_name}", key=f"preds_{model_name}"):
                display_predictions(pipeline, X_test, y_test, problem_type)
                
        except Exception as e:
            st.error(f"❌ Error with {model_name}: {str(e)}")
    
    return results

def encode_target(y):
    """Encode categorical target variables to numeric"""
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        return le.fit_transform(y), le
    return y, None

def display_model_metrics(data):
    """Main function to display model evaluation metrics"""
    with st.container():
        st.markdown("#### 🎯 Select Target Variable")
        target_col = st.selectbox(
            "Choose the column to predict:",
            data.columns,
            key="target_select"
        )
        
        if target_col:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            y_encoded, label_encoder = encode_target(y)
            
            problem_type = "Classification" if len(np.unique(y_encoded)) < 10 else "Regression"
            st.info(f"🔮 Detected: {problem_type} problem")
            
            with st.container():
                st.markdown("#### Select Models to Evaluate")
                selected_models = st.multiselect(
                    f"Choose {problem_type} models:",
                    list(MODEL_OPTIONS[problem_type].keys()),
                    default=["Random Forest", "XGBoost"],
                    key="model_select"
                )
            
            if selected_models:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42
                )
                
                results = evaluate_models(X, y_encoded, problem_type, selected_models)
                metrics_dict = {r["Model"]: r for r in results}
                
                if results:
                    results_df = pd.DataFrame(results).set_index("Model")
                    
                    # Display the raw metrics without percentage formatting
                    st.dataframe(
                        results_df.style.background_gradient(
                            cmap='Blues',
                            axis=None,
                            vmin=0,
                            vmax=1 if problem_type == "Classification" else None
                        ),
                        use_container_width=True,
                        height=min(400, 45 + 35*len(results))
                    )
                    
                    # Add formatted metrics as text annotations
                    formatted_df = results_df.copy()
                    if problem_type == "Classification":
                        formatted_df = formatted_df.applymap(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
                    else:
                        formatted_df = formatted_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                    
                    st.write("Formatted Metrics:")
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    metric = st.selectbox(
                        "Select metric to visualize:",
                        ["Accuracy", "Precision", "Recall", "F1 Score"] if problem_type == "Classification" 
                        else ["R² Score", "RMSE", "MAE"],
                        key="metric_select"
                    )
                    
                    fig = px.bar(
                        results_df.reset_index(),
                        x="Model",
                        y=metric,
                        color="Model",
                        color_discrete_sequence=px.colors.sequential.Blues_r,
                        text=results_df[metric].apply(
                            lambda x: f"{x:.1%}" if problem_type == "Classification" else f"{x:.3f}"
                        ),
                        height=400,
                        title=f"Model Comparison by {metric}"
                    )
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=60, b=20),
                        yaxis_title=metric,
                        xaxis_title=None,
                        uniformtext_minsize=8,
                        uniformtext_mode='hide'
                    )
                    fig.update_traces(
                        textposition='outside',
                        marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add download functionality
                    download_models(
                        selected_models, problem_type,
                        X_train, y_train, X_test, y_test,
                        metrics_dict,
                        target_col
                    )