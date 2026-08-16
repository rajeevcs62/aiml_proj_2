import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Page setup
st.set_page_config(page_title="BITS PROJECT", page_icon="", layout="wide")

#Added navbar
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #2C3E50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}
div.stButton > button:first-child:hover {
    background-color: #1ABC9C;
    color: white;
}
.navbar {
    display: flex;
    justify-content: space-around;
    background-color: #2C3E50;
    padding: 10px;
    border-radius: 8px;
}
.navbar a {
    color: white;
    text-decoration: none;
    font-weight: bold;
    font-size: 16px;
}
.navbar a:hover {
    color: #1ABC9C;
}
</style>

<div class="navbar">
    <a href="#ml_project">BITS ASSIGNMENT 2</a>
   
</div>
""", unsafe_allow_html=True)

# Intro section
st.header("PROJECT TOPIC:")
st.write("This app demonstrates multiple classification models on employee attrition done under IBM HR Analytics Employee Attrition dataset.")
st.write("MODEL DATA:https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv")

# Load dataset from GitHub raw URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
st.write("Sample Data of Model:", df.head())

# Encode categorical features
df_encoded = pd.get_dummies(df.drop("Attrition", axis=1))
y = df["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier() 
}

# Model selection
choice = st.selectbox("Choose a model", list(models.keys()))

if st.button("Train & Evaluate"):
    model = models[choice]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else preds

    st.write(f"### Results for {choice}")
    st.write("Accuracy:", accuracy_score(y_test, preds))
    st.write("AUC:", roc_auc_score(y_test, probs))
    st.write("Precision:", precision_score(y_test, preds))
    st.write("Recall:", recall_score(y_test, preds))
    st.write("F1 Score:", f1_score(y_test, preds))
    st.write("MCC:", matthews_corrcoef(y_test, preds))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"{choice} (AUC = {roc_auc_score(y_test, probs):.2f})")
    ax2.plot([0,1],[0,1],'k--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)
