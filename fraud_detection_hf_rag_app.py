import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from imblearn.over_sampling import SMOTE
import warnings
import json
import time
from datetime import datetime
import kagglehub

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Explainable Low-Resource Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-size: 2.5em;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .st-emotion-cache-1wv7c0w {
        border-radius: 10px;
        background-color: #f9f9f9;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stAlert {
        border-radius: 10px;
    }
    .stMetric {
        background-color: #e6f3ff;
        border-left: 5px solid #007bff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .st-emotion-cache-1wv7c0w h3, .st-emotion-cache-1g6x56q h3 {
        color: #004d99;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and model training
@st.cache_data
# Cache data loading and model training
@st.cache_data
def load_data():
    """Loads the dataset from a local file."""
    try:
        df = pd.read_csv("creditcardfraud.csv")
        df = df.drop_duplicates().copy()
        return df
    except FileNotFoundError:
        st.error("The 'creditcardfraud.csv' file was not found in the repository. Please upload the file and ensure it is in the same directory as the app.py script.")
        return None
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

@st.cache_resource
def train_model(df, model_type='Logistic Regression'):
    """Trains and optimizes the specified model."""
    start_time = time.time()
    
    # Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Oversampling with SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Custom scorer to prioritize recall
    def custom_scorer(y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        return (precision + 2 * recall) / 3

    my_scorer = make_scorer(custom_scorer)

    # Model training and optimization
    if model_type == 'Logistic Regression':
        model = LogisticRegression(solver='liblinear', random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }
    else:  # RandomForest
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_leaf': [1, 2]
        }
    
    grid_search = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=5), scoring=my_scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    optimized_model = grid_search.best_estimator_
    y_pred = optimized_model.predict(X_test_scaled)
    
    training_duration = time.time() - start_time
    
    return optimized_model, y_test, y_pred, X_test, scaler, training_duration, grid_search.best_params_

def create_confusion_matrix_plot(y_test, y_pred):
    """Creates an interactive confusion matrix plot."""
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual Legitimate', 'Actual Fraud'], columns=['Predicted Legitimate', 'Predicted Fraud'])
    
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion Matrix")
    fig.update_xaxes(side="top")
    return fig

def create_pr_curve_plot(model, X_test, y_test):
    """Creates a Precision-Recall curve plot."""
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[len(y_test[y_test==1]) / len(y_test), len(y_test[y_test==1]) / len(y_test)],
                             mode='lines', name='No-Skill Line', line=dict(dash='dash')))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    return fig

def create_shap_plot(model, X_test, scaler):
    """Generates and displays SHAP summary plot."""
    try:
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        
        # Display the SHAP summary plot
        st.subheader("💡 SHAP Feature Importance")
        st.write("This plot shows which features are most important for the model's predictions. The x-axis indicates the SHAP value, and each point is a transaction.")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
        
        # Display the SHAP dependence plot for the most important feature
        st.subheader("📊 SHAP Dependence Plot for Top Feature")
        top_feature_idx = np.argsort(np.mean(np.abs(shap_values.values), axis=0))[-1]
        top_feature_name = X_test.columns[top_feature_idx]
        st.write(f"This plot shows the effect of **{top_feature_name}** on the model's output.")
        fig, ax = plt.subplots()
        shap.dependence_plot(top_feature_idx, shap_values.values, X_test, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"SHAP plot generation failed. This can happen with certain models or data. Error: {e}")

# Main app function
def main():
    st.title("Explainable Fraud Detection")

    st.markdown("This application demonstrates a machine learning pipeline for detecting fraudulent transactions, with a focus on explainability. The model is trained on a small, anonymized dataset from Kaggle to simulate a low-resource environment.")

    with st.spinner("Loading and training model... This may take a few minutes."):
        df = load_data()
        if df is None:
            return

        model_choice = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "Random Forest Classifier"])
        
        optimized_model, y_test, y_pred, X_test, scaler, train_time, best_params = train_model(df, model_choice)
        
    st.sidebar.success("Model training complete!")
    st.sidebar.markdown(f"**Training time:** {train_time:.2f} seconds")

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Model Explainability", "Test a Transaction"])

    with tab1:
        st.header("Dashboard & Performance")
        st.subheader("Model Performance on Test Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        with col3:
            st.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
        
        st.subheader("Confusion Matrix")
        st.plotly_chart(create_confusion_matrix_plot(y_test, y_pred), use_container_width=True)
        
        st.subheader("Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

        st.subheader("Precision-Recall Curve")
        pr_curve_plot = create_pr_curve_plot(optimized_model, scaler.transform(X_test), y_test)
        st.plotly_chart(pr_curve_plot, use_container_width=True)

    with tab2:
        st.header("Model Explainability")
        
        st.info(f"Using {model_choice} with the best parameters: {best_params}", icon="ℹ️")

        st.subheader("Model Insights")
        
        insights_text = """
        Based on the trained model and feature analysis:
        
        1. **Time-based Patterns**: The model has learned that transactions during certain hours are more likely to be fraudulent. Anonymized feature analysis (V-features) often captures patterns related to the time of day.
        
        2. **Amount Anomalies**: Both very high transaction amounts and unusual amount patterns relative to typical spending are strong fraud indicators.
        
        3. **Behavioral Patterns**: The anonymized V-features capture intricate customer behavioral patterns, and certain combinations of these features are highly predictive of fraud.
        
        4. **Interaction Effects**: The model excels at detecting fraud when multiple risk factors combine (e.g., unusual amount + unusual time + specific behavioral patterns).
        
        5. **Feature Importance**: The most important features, as shown by the SHAP plot, are those that most significantly influence the model's decision-making process.
        """
        st.markdown(insights_text)

        create_shap_plot(optimized_model, X_test, scaler)

    with tab3:
        st.header("Test a New Transaction")
        st.markdown("Enter details for a hypothetical transaction to see if the model predicts it as fraudulent.")
        
        # Create a dictionary of input fields for the transaction
        transaction_data = {}
        for col in X_test.columns:
            if col == 'Time':
                transaction_data[col] = st.number_input(f"Enter {col} (seconds since first transaction)", min_value=0, value=150000)
            elif col == 'Amount':
                transaction_data[col] = st.number_input(f"Enter {col}", min_value=0.0, value=150.0)
            else:
                transaction_data[col] = st.number_input(f"Enter {col}", min_value=-30.0, max_value=30.0, value=0.0)

        if st.button("Predict Transaction Class"):
            # Create a DataFrame from the input data
            input_df = pd.DataFrame([transaction_data])
            
            # Scale the input data using the same scaler
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = optimized_model.predict(input_scaled)[0]
            prediction_proba = optimized_model.predict_proba(input_scaled)[0]
            
            st.subheader("Prediction Results")
            if prediction == 1:
                st.error("🚨 **This transaction is predicted to be FRAUDULENT.**")
            else:
                st.success("✅ **This transaction is predicted to be LEGITIMATE.**")
            
            st.info(f"Probability of being Legitimate: **{prediction_proba[0]:.4f}**")
            st.info(f"Probability of being Fraudulent: **{prediction_proba[1]:.4f}**")
            
            st.subheader("Why this prediction was made")
            
            # Use SHAP to explain the single prediction
            try:
                explainer = shap.Explainer(optimized_model, X_test)
                shap_values_single = explainer(input_df)

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(shap_values_single[0], show=False)
                st.pyplot(fig)

                st.markdown("The waterfall plot above shows how each feature contributes to the final prediction. Features that push the prediction towards fraud are shown in red, while those pushing it towards legitimate are in blue.")

            except Exception as e:
                st.error(f"SHAP explanation for this transaction failed. Error: {e}")

if __name__ == "__main__":
    main()
