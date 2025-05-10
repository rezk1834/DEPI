import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Page Config
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("heart_attack_dataset.csv")

data = load_data()

# Title
st.title("💓 Heart Attack Risk Prediction")
st.markdown("### DEPI Graduation Project")

# Sidebar Navigation
st.sidebar.title("🔧 Settings")
model_option = st.sidebar.selectbox("Select a Machine Learning Model", ["Random Forest", "Logistic Regression", "SVM"])
show_data = st.sidebar.checkbox("Show Raw Dataset", False)
show_corr = st.sidebar.checkbox("Show Correlation Heatmap", False)

# Display Dataset
st.markdown("## 🧾 Dataset Overview")
if show_data:
    st.dataframe(data)

# EDA Section
with st.expander("🔍 Explore Data Summary and Missing Values"):
    st.write("### Summary")
    st.write(data.describe())
    st.write("### Missing Values")
    st.write(data.isnull().sum())

if show_corr:
    st.markdown("### 🔗 Feature Correlation")
    corr = data.corr(numeric_only=True)
    fig_corr, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

# Preprocessing
X = data.drop(columns=["Treatment"])
y = data["Treatment"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection
if model_option == "Random Forest":
    model = RandomForestClassifier(random_state=42)
elif model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = SVC(probability=True)

# Train Model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluation Section
st.markdown("## 📊 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

with col2:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)

# ROC Curve
st.subheader("📈 ROC Curve")
y_probs = model.predict_proba(X_test_scaled)
if y_probs.shape[1] > 1:
    y_probs = y_probs[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label=model.classes_[1])
roc_auc = auc(fpr, tpr)

fig_roc, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="red")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic")
ax.legend()
st.pyplot(fig_roc)

# Footer
st.markdown("---")
st.markdown("**DEPI Graduation Project - Heart Attack Risk Classifier**")
