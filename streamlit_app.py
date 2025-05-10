import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Set Streamlit page config
st.set_page_config(page_title="Heart Attack Prediction", layout="wide")

# Title
st.title("💓 Heart Attack Prediction App")
st.caption("Graduation Project - DEPI")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("heart_attack_dataset.csv")

df = load_data()

# Display data
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# Data summary
with st.expander("📊 Dataset Information"):
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Null Values:**")
    st.dataframe(df.isnull().sum())
    st.write("**Descriptive Statistics:**")
    st.dataframe(df.describe(), use_container_width=True)

# Prepare features and target
if "Treatment" not in df.columns:
    st.error("Missing 'Treatment' column in the dataset.")
    st.stop()

X = df.drop(columns=['Treatment'])
y = df['Treatment']

# Convert categorical if needed
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar model selection
st.sidebar.header("⚙️ Model Selection")
model_name = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "Support Vector Machine"])

# Model instantiation
if model_name == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = SVC(probability=True)

# Train the model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

# Layout columns for results
col1, col2 = st.columns(2)

# Classification report (styled)
with col1:
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    styled_report = report_df.style.background_gradient(cmap="YlGnBu", axis=1)
    st.dataframe(styled_report, use_container_width=True)

# Confusion Matrix
with col2:
    st.subheader("🧮 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# ROC Curve
st.subheader("📈 ROC Curve")
if len(np.unique(y)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("ROC curve is only available for binary classification.")
