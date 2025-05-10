import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# App title
st.title("Heart Attack Prediction App")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("heart_attack_dataset.csv")

Health = load_data()

st.subheader("Dataset Preview")
st.dataframe(Health.head())

st.subheader("Column Names")
st.write(Health.columns.tolist())

st.subheader("Null Values")
st.write(Health.isnull().sum())

st.subheader("Data Summary")
st.write(Health.describe())

# Drop rows with missing values for simplicity
Health.dropna(inplace=True)

# Feature and target
X = Health.drop('Treatment', axis=1)
y = Health['Treatment']

# Identify numeric and categorical features
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Sidebar - Model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "Support Vector Machine"])

if model_option == "Random Forest":
    model = RandomForestClassifier()
elif model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_option == "Support Vector Machine":
    model = SVC(probability=True)

# Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
y_probs = pipeline.predict_proba(X_test)

# Evaluation
st.subheader("Model Evaluation")
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
st.pyplot(fig)

# ROC Curve (for multiclass: average only one-vs-rest logic)
st.subheader("ROC Curve")
fig, ax = plt.subplots()
if len(np.unique(y)) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
else:
    for i in range(y_probs.shape[1]):
        fpr, tpr, _ = roc_curve(y_test == model.classes_[i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model.classes_[i]} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)
