import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Helper functions
def generate_credit_score():
    return np.clip(np.random.normal(650, 70), 300, 850)

def generate_income():
    return np.round(np.random.normal(50000, 20000), -2)

def determine_creditworthy(credit_score, dti, late_payments, bankruptcies):
    if credit_score > 700 and dti < 0.35 and late_payments == 0 and bankruptcies == 0:
        return 1
    elif credit_score < 550 or dti > 0.6 or bankruptcies > 0:
        return 0
    else:
        return np.random.choice([0, 1], p=[0.3, 0.7])

def estimate_approved_amount(income, credit_score, is_creditworthy):
    if is_creditworthy:
        base = income * 0.3
        multiplier = credit_score / 850
        return int(base * multiplier)
    return 0

# Categories
genders = ['Male', 'Female', 'Other']
marital_statuses = ['Single', 'Married', 'Divorced']
education_levels = ['High School', 'Bachelors', 'Masters', 'PhD']
employment_statuses = ['Employed', 'Self-employed', 'Unemployed', 'Retired']
loan_purposes = ['Car', 'Home', 'Education', 'Business', 'Personal']

# Generate data
n_samples = 1000
data = []

for _ in range(n_samples):
    age = np.random.randint(18, 70)
    gender = random.choice(genders)
    marital_status = random.choice(marital_statuses)
    education = random.choice(education_levels)
    employment = random.choice(employment_statuses)
    dependents = np.random.randint(0, 5)

    income = generate_income()
    monthly_income = income / 12
    savings = np.random.uniform(500, income * 0.5)
    checking = np.random.uniform(100, income * 0.3)
    dti = np.clip(np.random.normal(0.3, 0.15), 0, 1)

    credit_score = int(generate_credit_score())
    credit_lines = np.random.randint(1, 10)
    open_lines = np.random.randint(1, credit_lines + 1)
    utilization = np.clip(np.random.beta(2, 5), 0, 1)
    late_payments = np.random.poisson(0.5)
    delinq_accounts = np.random.binomial(1, 0.1)
    bankruptcies = np.random.binomial(1, 0.05)

    requested_amount = int(np.random.uniform(5000, 50000))
    loan_term = random.choice([12, 24, 36, 48, 60])
    interest_rate = round(np.clip(np.random.normal(10, 3), 3, 25), 2)
    loan_purpose = random.choice(loan_purposes)

    is_creditworthy = determine_creditworthy(credit_score, dti, late_payments, bankruptcies)
    approved_amount = estimate_approved_amount(income, credit_score, is_creditworthy)

    data.append([
        age, gender, marital_status, education, employment, dependents,
        income, monthly_income, dti, savings, checking,
        credit_score, credit_lines, open_lines, utilization,
        late_payments, delinq_accounts, bankruptcies, loan_purpose,
        requested_amount, loan_term, interest_rate,
        is_creditworthy, approved_amount
    ])

# Define column names
columns = [
    "age", "gender", "marital_status", "education_level", "employment_status", "number_of_dependents",
    "annual_income", "monthly_income", "debt_to_income_ratio", "savings_account_balance", "checking_account_balance",
    "credit_score", "number_of_credit_lines", "number_of_open_credit_lines", "credit_utilization_ratio",
    "number_of_late_payments", "delinquent_accounts", "bankruptcies", "loan_purpose",
    "requested_loan_amount", "loan_term", "interest_rate",
    "is_creditworthy", "approved_loan_amount"
]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Keep only the most relevant columns for prediction
important_columns = [
    "annual_income", "marital_status", "education_level", "number_of_dependents",
    "credit_score", "debt_to_income_ratio", "requested_loan_amount",
    "is_creditworthy", "approved_loan_amount"
]

df_reduced = df[important_columns]

# Label encoding for categorical features
label_encoders = {}
for column in ["marital_status", "education_level"]:
    le = LabelEncoder()
    df_reduced[column] = le.fit_transform(df_reduced[column])
    label_encoders[column] = le

# Split the data into features and targets
X = df_reduced[["annual_income", "marital_status", "education_level", "number_of_dependents"]]
y = df_reduced.drop(columns=["annual_income", "marital_status", "education_level", "number_of_dependents"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model for predicting continuous variables (e.g., credit_score, requested_loan_amount)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train[["credit_score", "debt_to_income_ratio", "requested_loan_amount", "approved_loan_amount"]])

# Model for predicting classification variable (e.g., is_creditworthy)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train["is_creditworthy"])

# Save models and label encoders
with open("credit_prediction_model.pkl", "wb") as model_file:
    pickle.dump(regressor, model_file)
    pickle.dump(classifier, model_file)

with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)

print("Models and label encoders saved as 'credit_prediction_model.pkl' and 'label_encoders.pkl'.")
