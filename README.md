# 🏦 Loan Default Prediction using Machine Learning

## 📌 Project Overview
This project applies **predictive analytics** to identify the likelihood of loan default using machine learning models. The goal is to assist financial institutions in **reducing credit risk** while ensuring **fair and efficient loan approvals**.

## 🎯 Objectives
- Build a classification model to predict **loan default** (1 = default, 0 = repaid).
- Compare traditional models with advanced ensemble methods.
- Recommend the most effective model based on **accuracy, recall, and precision**.
- Provide business insights on key features influencing loan defaults.

## 📂 Dataset
- **Source**: Public loan dataset (5,960 records, 12 input variables)  
- **Target Variable**: `BAD` (1 = default, 0 = repaid)  
- **Key Features**:
  - `LOAN`: Loan amount approved  
  - `MORTDUE`: Outstanding mortgage balance  
  - `VALUE`: Current property value  
  - `DEBTINC`: Debt-to-income ratio  
  - `JOB`, `REASON`, `YOJ`, `DEROG`, `DELINQ`, etc.  

## ⚙️ Methodology
1. Data cleaning & handling missing values
2. Exploratory Data Analysis (EDA)
3. Feature engineering & encoding categorical variables
4. Model training & hyperparameter tuning
5. Performance evaluation using **Accuracy, Precision, Recall, F1-score**

## 🤖 Models Evaluated
- Logistic Regression  
- Decision Tree (tuned & default)  
- Random Forest (tuned & default)  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  
- KNN, Naive Bayes  

## 📊 Results Summary
- **XGBoost**: Best overall performer – 92.4% accuracy, 71% recall, 89% precision    
- **LightGBM**: High accuracy (92.1%), recall (70%), precision (88%)   
- **CatBoost**: Highest test precision (89.7%) with solid recall (68%)   
- **Tuned Decision Tree**: Best recall (73.9%) – most suitable when **missing a defaulter is costly**   

✅ **Best Balanced Models**: XGBoost, LightGBM, CatBoost  
❌ **Models to Avoid**: Logistic Regression, KNN, Naive Bayes 

## 📌 Key Business Insights
- **Debt-to-Income Ratio (DEBTINC)** and **Delinquency History** are strong predictors of default .  
- A high DEBTINC (>43.7) significantly increases the probability of default.  
- Employment type (`JOB`) and loan purpose (`REASON`) also influence default likelihood.

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)  
- XGBoost, LightGBM, CatBoost  
- Jupyter Notebook  

## 🚀 Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ivikassingh2/default/blob/main/Predictive_Analytics_Loan_Default.ipynb)

## 📊 Model Performance Comparison

| Model               | Train Acc | Test Acc | Train Recall | Test Recall | Train Precision | Test Precision | Train F1 | Test F1 |
|---------------------|-----------|----------|--------------|-------------|-----------------|----------------|----------|---------|
| **Tuned Random Forest** | 1.000 | 0.908 | 1.000 | 0.683 | 1.000 | 0.824 | 1.000 | 0.747 |
| Random Forest       | 1.000 | 0.904 | 1.000 | 0.641 | 1.000 | 0.839 | 1.000 | 0.727 |
| Tuned Decision Tree | 0.871 | 0.858 | 0.792 | 0.739 | 0.643 | 0.621 | 0.710 | 0.675 |
| Decision Tree       | 1.000 | 0.880 | 1.000 | 0.625 | 1.000 | 0.736 | 1.000 | 0.676 |
| Logistic Regression | 0.812 | 0.809 | 0.093 | 0.073 | 0.733 | 0.722 | 0.164 | 0.132 |
| Gradient Boosting   | 0.945 | 0.908 | 0.791 | 0.653 | 0.920 | 0.850 | 0.851 | 0.739 |
| **XGBoost**         | 0.999 | **0.924** | 0.998 | 0.706 | 1.000 | 0.894 | 0.999 | **0.789** |
| LightGBM            | 0.994 | 0.921 | 0.972 | 0.700 | 0.995 | 0.880 | 0.984 | 0.780 |
| CatBoost            | 0.972 | 0.921 | 0.882 | 0.681 | 0.977 | **0.897** | 0.927 | 0.774 |
| KNN                 | 0.858 | 0.804 | 0.362 | 0.218 | 0.829 | 0.520 | 0.504 | 0.308 |
| Naive Bayes         | 0.823 | 0.814 | 0.189 | 0.132 | 0.707 | 0.671 | 0.298 | 0.220 |

## 📊 Model Performance Interpretation - Bank Loan Default Prediction

| Model | Interpretation Summary |
|-------|------------------------|
| **Tuned Random Forest** | Overfitting is evident — 100% train accuracy, but test recall is only **68%**. High train precision and recall indicate it's memorizing training data. Test precision is decent (**82%**), but generalization may be limited. |
| **Random Forest (Default)** | Similar to the tuned version — again overfitting. Perfect training performance, but test recall is **64%**. Despite good test precision (**84%**), it still misses many defaulters. |
| **Tuned Decision Tree** | Better generalization than random forests. Train accuracy is **87%**, test accuracy **86%**. Test recall (**74%**) shows good ability to catch defaulters, though test precision is lower (**62%**), meaning more false positives. |
| **Decision Tree (Default)** | Clear overfitting — 100% training metrics, but test recall is only **62%**, and test precision is **74%**. Pruning or parameter tuning is needed to reduce complexity. |
| **Logistic Regression** | Poor model for this task. Extremely low test recall (**7%**) means it misses nearly all defaulters. Accuracy (~81%) is misleading due to class imbalance. Not recommended as-is. |
| **Gradient Boosting** | Strong performer: test accuracy **90.7%**, recall **65%**, precision **85%**. Very good balance between bias and variance. A reliable and generalizable model. |
| **XGBoost** | One of the top performers. Excellent test accuracy (**92.4%**), strong recall (**71%**), and precision (**89%**). Some overfitting, but still generalizes well. |
| **LightGBM** | High-performing and efficient. Test accuracy (**92.1%**), recall (**70%**), and precision (**88%**) are all impressive. Great choice for deployment due to speed and accuracy. |
| **CatBoost** | Excellent all-rounder. Test accuracy (**92.1%**), recall (**68%**), and highest test precision (**89.7%**). Handles categorical variables well, and balances precision-recall tradeoffs smartly. |
| **KNeighborsClassifier** | Weakest performer. Test recall only **22%** and F1 score **31%**. Struggles with class imbalance and may be unsuitable for high-dimensional data. |
| **Naive Bayes** | Low-performing model. Test recall is just **13%**, and F1 score is **22%**. Not effective for identifying defaulters. |

---

### 🏆 Recommendation

#### ✅ Best Performing Models (Balanced):
- **CatBoost** – Best precision, solid recall, highest test accuracy  
- **XGBoost** – Very high recall and precision, strong test performance  
- **LightGBM** – Fast, accurate, and interpretable  

#### ❌ Models to Avoid:
- **Logistic Regression** – Fails on recall, unsuitable unless rebalanced  
- **KNN** – Underperforms across key metrics  
- **Naive Bayes** – Misses most defaulters, low F1  

---

### ⚖️ Overfitting Watchlist
- **Tuned & Default Random Forest** – Perfect train performance, but lower test recall  
- **Default Decision Tree** – Overfits with poor test generalization  

