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
from sklearn.ensemble import RandomForestClassifier

st.title("Classification Models on Adult Income Dataset")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]
    df = pd.read_csv(url, header=None, names=columns, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    return df

df = load_data()
st.write("Dataset Preview:", df.head())

# Encode categorical features
df_encoded = pd.get_dummies(df.drop("income", axis=1))
y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
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
