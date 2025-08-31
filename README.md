# Credit-Prediction-Project

## 📌 Project Overview

The **Credit Prediction Project** is a machine learning-based system designed to predict whether an individual is a good or bad credit risk. This type of analysis is vital for financial institutions to make informed decisions when offering credit or loans to customers. The project uses a dataset containing various customer attributes such as credit history, employment status, loan amount, and more, to train classification models capable of making accurate creditworthiness predictions.

## 🎯 Objective

To build and evaluate multiple machine learning models that can classify whether a person is likely to default or fulfill credit obligations. The aim is to identify the best-performing model and optimize it for better accuracy and reliability in credit risk assessment.

---

## 🧰 Technologies Used

- **Python**
- **Pandas & NumPy** – Data manipulation
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning modeling and evaluation
- **Jupyter Notebook** – Interactive development and exploration

---

## 📊 Dataset

- The dataset used is the **German Credit Dataset**, which includes 1000 samples with 20 features.
- The target variable is `Creditability`, where:
  - `1` indicates good credit
  - `0` indicates bad credit

Key features include:
- Duration of credit
- Credit amount
- Age
- Job type
- Housing status
- Foreign worker status
- Number of existing credits, etc.

---

## 🔍 Project Workflow

1. **Data Loading & Exploration**  
   - Loaded the dataset using `pandas`
   - Performed exploratory data analysis (EDA)
   - Visualized distributions, correlations, and categorical features

2. **Data Preprocessing**  
   - Handled missing values
   - Converted categorical variables using one-hot encoding or label encoding
   - Scaled numerical features using `StandardScaler`

3. **Model Building**  
   Evaluated and compared several classification models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Gradient Boosting Classifier

4. **Model Evaluation**  
   Used metrics like:
   - Accuracy
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)

5. **Model Comparison & Conclusion**  
   - Compared models to find the best one in terms of accuracy and generalization
   - Gradient Boosting or Random Forest showed strong performance (based on the code)

---

## ✅ Results

The machine learning models, especially ensemble techniques like **Random Forest** and **Gradient Boosting**, showed high accuracy and performed well in distinguishing between good and bad credit risks. The project demonstrates the effectiveness of ML in credit scoring applications.

---

## 📁 Repository Structure

```
Credit-Prediction-Project/
│
├── Credit_Prediction.ipynb      # Main Jupyter notebook with code and analysis
├── german_credit_data.csv       # Dataset used for modeling
├── README.md                    # Project description and documentation
```

---

## 📌 Future Work

- Implement hyperparameter tuning with GridSearchCV
- Deploy the best model using Flask or Streamlit
- Use cross-validation for more robust evaluation
- Apply feature selection techniques
- Test with real-time or larger credit datasets

---

## 👤 Author

**Deba-Nick**  
GitHub: [@Deba-Nick](https://github.com/Deba-Nick)\n
**MahamadSahjad**
GitHub: [@MahamadSahjad](https://github.com/MahamadSahjad)
---




