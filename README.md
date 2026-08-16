# ML PROJECT 2: Machine Learning Classification Project 

**Steamlit URL: https://aimlproj2-rajeevcs62.streamlit.app/**

## a. Problem Statement
The goal of this project is to apply multiple machine learning classification models to analyse employee attrition over data IBM HR Analytics EMployee Attrition Dataset using different methods mentioned below. 

---

## b. Dataset Description
- **Dataset Name:** IBM HR Analytics Employee Attrition Dataset  
- **Source:** Kaggle (IBM HR Analytics Employee Attrition & Performance)  
- **Rows:** ~1,470  
- **Features:** 35+ (mix of categorical and numerical)  
- **Target Variable:** Attrition (Yes/No)  

---

## c. GitHub Repository Link:(https://github.com/rajeevcs62/aiml_proj_2)

---

## d. Models Link: https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv

##Model method:
- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- Support Vector Machine (SVM) / Gradient Boosting (depending on your choice for the 6th model)

### Comparison Table of Metrics

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1   | MCC   |
|----------------------|----------|-------|-----------|--------|------|-------|
| Logistic Regression  | 0.8878   | 0.7665| 0.75      | 0.2308 | 0.3529 | 0.3754 |
| Decision Tree        | 0.7823   | 0.5379| 0.1951    | 0.2051 | 0.2000 | 0.0741 |
| kNN                  | 0.8537   | 0.5275| 0.3571    | 0.1282 | 0.1887 | 0.1480 |
| Naive Bayes          | 0.7721   | 0.7371| 0.3056    | 0.5641 | 0.3964 | 0.2903 |
| Random Forest        | 0.8707   | 0.7278| 0.5714    | 0.1026 | 0.1739 | 0.2020 |

---

## e. Observations on Model Performance

| ML Model Name        | Observation about model performance |
|----------------------|-------------------------------------|
| Logistic Regression  | High accuracy and AUC, but recall is low, meaning it misses many attrition cases. |
| Decision Tree        | Lower accuracy and weak precision/recall balance; tends to overfit. |
| kNN                  | Moderate accuracy, but recall is poor; struggles with imbalanced data. |
| Naive Bayes          | Lower accuracy but strong recall; identifies more attrition cases, though precision is weaker. |
| Random Forest        | Good accuracy, but recall is very low; precision is decent. |
| **Overall Winner**   | Logistic Regression — best balance of accuracy, AUC, and MCC among the five models. |

---

### Notes
- Accuracy alone is not sufficient; AUC, Precision, Recall, F1, and MCC provide deeper insights.  
- Imbalanced datasets require careful attention to Recall and Precision.  
- Ensemble methods (Random Forest) often outperform single models.  

---
