# Machine Learning Assignment 2

## a. Problem Statement
The objective is to implement, evaluate, and deploy multiple classification models to accurately predict outcomes on a chosen dataset. The solution involves building an interactive Streamlit web application to serve pre-trained models and calculate evaluation metrics dynamically on uploaded test data.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Features**: 30 (Measurements of cell nuclei characteristics computed from a digitized image of a fine needle aspirate of a breast mass).
- **Instances**: 569
- **Target**: Binary classification (Malignant vs. Benign).

## c. Github Repository Link
[Insert your GitHub Repository Link here]

## d. Models Used
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| KNN | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Support Vector Machine| 0.9561 | 0.9974 | 0.9714 | 0.9577 | 0.9645 | 0.9072 |

## Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Achieved the highest overall performance across almost all metrics (Accuracy 97.37%, MCC 0.9439). The dataset's features, once scaled, are highly linearly separable, making this model highly effective. |
| Decision Tree | Showed the lowest AUC (0.9440) and joint-lowest accuracy. This suggests a slight tendency to overfit the training data compared to smoother, probabilistic models on this specific dataset. |
| KNN | Tied with the Decision Tree in Accuracy (94.74%) but achieved a much better AUC (0.9820), showing that while it makes some absolute classification errors, its probability ranking is strong. |
| Naive Bayes | Performed exceptionally well (96.49% Accuracy, 0.9974 AUC) despite the assumption of feature independence. It tied for the highest recall (98.59%), making it excellent at catching true positives. |
| Support Vector Machine| Delivered very strong results (95.61% Accuracy, 0.9974 AUC). While slightly trailing Logistic Regression and Naive Bayes in F1 score, it handled the high-dimensional feature space effectively. |
| **Overall, Winner for your dataset?** | **Logistic Regression**. It outperformed the others by achieving the highest Accuracy, F1 Score, and MCC, proving to be the most reliable model for distinguishing between benign and malignant cases in this dataset. |# ML_Assignment_2
