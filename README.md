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

## 🚀 How to Run
```bash
git clone https://github.com/ivikassingh2/Default.git
cd loan-default-prediction
pip install -r requirements.txt
jupyter notebook Predictive_Analytics_Loan_Default.ipynb
