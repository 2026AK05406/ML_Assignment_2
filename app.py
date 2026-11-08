import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.title("Machine Learning Classifier Dashboard")
st.markdown("Upload test data to evaluate 5 pre-trained models on the UCI Breast Cancer dataset.")

# Model Selection Dropdown
model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

# Dataset Upload Option
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())
    
    # Split features and target
    if 'target' not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X_test = data.drop(columns=['target'])
        y_test = data['target']
        
        # Load Model
        filename = f'model/{model_name.replace(" ", "_")}.pkl'
        model = joblib.load(filename)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.write("### Evaluation Metrics")
        metrics_df = pd.DataFrame({
            "Accuracy": [f"{acc:.4f}"],
            "AUC Score": [f"{auc:.4f}"],
            "Precision": [f"{prec:.4f}"],
            "Recall": [f"{rec:.4f}"],
            "F1 Score": [f"{f1:.4f}"],
            "MCC": [f"{mcc:.4f}"]
        })
        st.table(metrics_df)
        
        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
