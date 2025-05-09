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

# Load Data
st.title("Heart Attack Prediction App")

@st.cache_data

def load_data():
    data = pd.read_csv("heart_attack_dataset.csv")
    return data

Health = load_data()
st.subheader("Dataset Preview")
st.dataframe(Health.head())

# Display Column Names
st.subheader("Column Names")
st.write(Health.columns.tolist())

# Null values
st.subheader("Null Values")
st.write(Health.isnull().sum())

# Describe data
st.subheader("Data Summary")
st.write(Health.describe())

# Feature and Target selection
X = Health.drop(columns=['target'])
y = Health['target']

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "Support Vector Machine"])

if model_option == "Random Forest":
    model = RandomForestClassifier()
elif model_option == "Logistic Regression":
    model = LogisticRegression()
elif model_option == "Support Vector Machine":
    model = SVC(probability=True)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
st.subheader("Model Evaluation")
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
y_probs = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)
