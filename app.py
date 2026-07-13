import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("heart_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'heart_cleaned.csv' is in the same directory.")
        return None

@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in categorical_cols:
        if col not in ['dataset']:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            le_dict[col] = le
    
    bool_cols = df_processed.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_processed[col] = df_processed[col].astype(int)
    
    df_processed = df_processed.drop(columns=['id', 'dataset'], errors='ignore')
    
    return df_processed, le_dict

@st.cache_resource
def train_all_models(df):
    df_processed, le_dict = preprocess_data(df)
    
    X = df_processed.drop(columns=['num'])
    y = df_processed['num'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    lr_metrics = {
        "accuracy": accuracy_score(y_test, lr_pred),
        "precision": precision_score(y_test, lr_pred),
        "recall": recall_score(y_test, lr_pred),
        "f1": f1_score(y_test, lr_pred),
        "roc_auc": roc_auc_score(y_test, lr_pred_proba),
        "conf_matrix": confusion_matrix(y_test, lr_pred),
        "y_test": y_test,
        "y_pred": lr_pred,
        "y_pred_proba": lr_pred_proba
    }
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    rf_metrics = {
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred),
        "recall": recall_score(y_test, rf_pred),
        "f1": f1_score(y_test, rf_pred),
        "roc_auc": roc_auc_score(y_test, rf_pred_proba),
        "conf_matrix": confusion_matrix(y_test, rf_pred),
        "y_test": y_test,
        "y_pred": rf_pred,
        "y_pred_proba": rf_pred_proba,
        "feature_importance": rf_model.feature_importances_
    }
    
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    xgb_metrics = {
        "accuracy": accuracy_score(y_test, xgb_pred),
        "precision": precision_score(y_test, xgb_pred),
        "recall": recall_score(y_test, xgb_pred),
        "f1": f1_score(y_test, xgb_pred),
        "roc_auc": roc_auc_score(y_test, xgb_pred_proba),
        "conf_matrix": confusion_matrix(y_test, xgb_pred),
        "y_test": y_test,
        "y_pred": xgb_pred,
        "y_pred_proba": xgb_pred_proba,
        "feature_importance": xgb_model.feature_importances_
    }
    
    return {
        "lr": {"model": lr_model, "metrics": lr_metrics},
        "rf": {"model": rf_model, "metrics": rf_metrics},
        "xgb": {"model": xgb_model, "metrics": xgb_metrics},
        "scaler": scaler,
        "feature_cols": X.columns,
        "le_dict": le_dict,
        "X_test": X_test_scaled
    }

df = load_data()
if df is not None:
    models_data = train_all_models(df)

st.sidebar.title("🩺 Navigation Menu")
menu = st.sidebar.radio(
    "Select a page:",
    ("Home (Prediction)", "Exploratory Data Analysis", "Model Comparison", "Model Details")
)

st.sidebar.markdown("---")
st.sidebar.info("📚 **Foundation of Data Science Project**\n\n"
                "Heart Disease Prediction System\n\n"
                "Models: LR | RF | XGB")


if menu == "Home (Prediction)":
    st.title("❤️ Heart Disease Prediction System")
    st.write("Enter patient clinical attributes to predict heart disease risk using multiple ML models.")
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Patient Vitals")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
            trestbps = st.number_input("Resting BP (mmHg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
            thalch = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
            
        with col2:
            st.subheader("Additional Tests")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
            exang_val = 1 if exang == "Yes" else 0
            
        with col3:
            st.subheader("Classification")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia Type", options=[0, 1, 2, 3])
            ca = st.selectbox("Major Vessels Count", options=[0, 1, 2, 3, 4])
            fbs = st.selectbox("Fasting Blood Sugar > 120?", options=["No", "Yes"])
            fbs_val = 1 if fbs == "Yes" else 0
            
        if st.button("🔮 Predict with All Models", type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                'age': [age],
                'trestbps': [trestbps],
                'chol': [chol],
                'thalch': [thalch],
                'oldpeak': [oldpeak],
                'cp': [cp],
                'exang': [exang_val],
                'thal': [thal],
                'ca': [ca],
                'fbs': [fbs_val]
            })
  
            for feat in models_data["feature_cols"]:
                if feat not in input_data.columns:
                    input_data[feat] = 0
            
            input_data = input_data[models_data["feature_cols"]]

            input_scaled = models_data["scaler"].transform(input_data)

            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                lr_pred = models_data["lr"]["model"].predict(input_scaled)[0]
                lr_prob = models_data["lr"]["model"].predict_proba(input_scaled)[0][1]
                st.metric("Logistic Regression", f"{lr_prob:.2%}", delta="High Risk" if lr_pred == 1 else "Low Risk")
                if lr_pred == 1:
                    st.warning("⚠️ Risk Detected")
                else:
                    st.success("✅ Low Risk")
            
            with col2:
                rf_pred = models_data["rf"]["model"].predict(input_scaled)[0]
                rf_prob = models_data["rf"]["model"].predict_proba(input_scaled)[0][1]
                st.metric("Random Forest", f"{rf_prob:.2%}", delta="High Risk" if rf_pred == 1 else "Low Risk")
                if rf_pred == 1:
                    st.warning("⚠️ Risk Detected")
                else:
                    st.success("✅ Low Risk")
            
            with col3:
                xgb_pred = models_data["xgb"]["model"].predict(input_scaled)[0]
                xgb_prob = models_data["xgb"]["model"].predict_proba(input_scaled)[0][1]
                st.metric("XGBoost", f"{xgb_prob:.2%}", delta="High Risk" if xgb_pred == 1 else "Low Risk")
                if xgb_pred == 1:
                    st.warning("⚠️ Risk Detected")
                else:
                    st.success("✅ Low Risk")
            
            st.markdown("---")
            st.info("💡 **Interpretation:** Multiple model consensus improves prediction reliability. "
                   "Consult a healthcare professional for medical decisions.")


elif menu == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    if df is not None:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Total Features", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Disease Cases", (df['num'] > 0).sum())
        
        st.markdown("---")
        
        with st.expander("📋 View Dataset Preview"):
            st.dataframe(df.head(10))
        
        with st.expander("📈 Summary Statistics"):
            st.dataframe(df.describe())
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Heart Disease Distribution")
            disease_counts = df['num'].apply(lambda x: "Disease" if x > 0 else "No Disease").value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#FF6B6B', '#4ECDC4']
            ax.pie(disease_counts.values, labels=disease_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors)
            ax.set_title("Disease vs No Disease", fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Age Distribution by Disease Status")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df, x='age', hue=df['num'].apply(lambda x: "Disease" if x > 0 else "No Disease"),
                        kde=True, ax=ax, palette=['#4ECDC4', '#FF6B6B'])
            ax.set_title("Age Distribution", fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("Feature Distributions")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df, x='chol', kde=True, ax=ax, color='skyblue')
            ax.set_title("Cholesterol Distribution", fontsize=12, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data=df, x='trestbps', kde=True, ax=ax, color='lightcoral')
            ax.set_title("Resting Blood Pressure Distribution", fontsize=12, fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("Feature Correlation Matrix")

        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, 
                   cbar_kws={"label": "Correlation"})
        ax.set_title("Correlation Heatmap of Numeric Features", fontsize=14, fontweight='bold')
        st.pyplot(fig)


elif menu == "Model Comparison":
    st.title("📊 Model Comparison & Performance")
    
    if df is not None:
        st.subheader("Performance Metrics Comparison")
        
        comparison_data = {
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Accuracy': [
                f"{models_data['lr']['metrics']['accuracy']:.4f}",
                f"{models_data['rf']['metrics']['accuracy']:.4f}",
                f"{models_data['xgb']['metrics']['accuracy']:.4f}"
            ],
            'Precision': [
                f"{models_data['lr']['metrics']['precision']:.4f}",
                f"{models_data['rf']['metrics']['precision']:.4f}",
                f"{models_data['xgb']['metrics']['precision']:.4f}"
            ],
            'Recall': [
                f"{models_data['lr']['metrics']['recall']:.4f}",
                f"{models_data['rf']['metrics']['recall']:.4f}",
                f"{models_data['xgb']['metrics']['recall']:.4f}"
            ],
            'F1-Score': [
                f"{models_data['lr']['metrics']['f1']:.4f}",
                f"{models_data['rf']['metrics']['f1']:.4f}",
                f"{models_data['xgb']['metrics']['f1']:.4f}"
            ],
            'ROC-AUC': [
                f"{models_data['lr']['metrics']['roc_auc']:.4f}",
                f"{models_data['rf']['metrics']['roc_auc']:.4f}",
                f"{models_data['xgb']['metrics']['roc_auc']:.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            accuracies = [
                models_data['lr']['metrics']['accuracy'],
                models_data['rf']['metrics']['accuracy'],
                models_data['xgb']['metrics']['accuracy']
            ]
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(['Logistic\nRegression', 'Random\nForest', 'XGBoost'], accuracies,
                         color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_ylim([0, 1])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.subheader("F1-Score Comparison")
            f1_scores = [
                models_data['lr']['metrics']['f1'],
                models_data['rf']['metrics']['f1'],
                models_data['xgb']['metrics']['f1']
            ]
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(['Logistic\nRegression', 'Random\nForest', 'XGBoost'], f1_scores,
                         color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
            ax.set_ylabel("F1-Score", fontsize=12)
            ax.set_ylim([0, 1])
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("ROC Curves - Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models_list = [
            ('Logistic Regression', models_data['lr']),
            ('Random Forest', models_data['rf']),
            ('XGBoost', models_data['xgb'])
        ]
        
        colors_list = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        for (model_name, model_info), color in zip(models_list, colors_list):
            fpr, tpr, _ = roc_curve(model_info['metrics']['y_test'], model_info['metrics']['y_pred_proba'])
            auc = model_info['metrics']['roc_auc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2.5, color=color)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


elif menu == "Model Details":
    st.title("📋 Detailed Model Analysis")
    
    if df is not None:
        model_choice = st.selectbox(
            "Select Model to Analyze:",
            options=['Logistic Regression', 'Random Forest', 'XGBoost']
        )
        
        model_key = {'Logistic Regression': 'lr', 'Random Forest': 'rf', 'XGBoost': 'xgb'}[model_choice]
        selected_model_data = models_data[model_key]
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_values = selected_model_data['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics_values['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics_values['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics_values['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics_values['f1']:.4f}")
        with col5:
            st.metric("ROC-AUC", f"{metrics_values['roc_auc']:.4f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(metrics_values['conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={"label": "Count"})
            ax.set_xlabel("Predicted Label", fontsize=11)
            ax.set_ylabel("True Label", fontsize=11)
            ax.set_xticklabels(['No Disease', 'Disease'])
            ax.set_yticklabels(['No Disease', 'Disease'])
            ax.set_title(f'{model_choice} - Confusion Matrix', fontsize=12, fontweight='bold')
            st.pyplot(fig)

        with col2:
            if model_key in ['rf', 'xgb']:
                st.subheader("Feature Importance")
                importances = selected_model_data['metrics']['feature_importance']
                feat_imp_df = pd.DataFrame({
                    'Feature': models_data['feature_cols'],
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#4ECDC4')
                ax.set_xlabel("Importance Score", fontsize=11)
                ax.set_title(f'{model_choice} - Top 10 Features', fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.subheader("Model Coefficients")
                coefficients = selected_model_data['model'].coef_[0]
                coef_df = pd.DataFrame({
                    'Feature': models_data['feature_cols'],
                    'Coefficient': coefficients
                }).sort_values(by='Coefficient', key=abs, ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in coef_df['Coefficient']]
                ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
                ax.set_xlabel("Coefficient Value", fontsize=11)
                ax.set_title(f'{model_choice} - Top 10 Coefficients', fontsize=12, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.invert_yaxis()
                st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("Detailed Classification Report")
        y_test = metrics_values['y_test']
        y_pred = metrics_values['y_pred']
        
        report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
