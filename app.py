from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the models and label encoders
with open('credit_prediction_model.pkl', 'rb') as model_file:
    regressor = pickle.load(model_file)
    classifier = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

app = Flask(__name__)

@app.route('/')
def index():
    # Render the index.html file from the templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Prepare input data
    annual_income = float(data['annual_income'])
    marital_status = label_encoders['marital_status'].transform([data['marital_status']])[0]
    education_level = label_encoders['education_level'].transform([data['education_level']])[0]
    number_of_dependents = int(data['number_of_dependents'])

    # Create feature vector
    X = np.array([[annual_income, marital_status, education_level, number_of_dependents]])

    # Make predictions
    prediction = regressor.predict(X)
    is_creditworthy = classifier.predict(X)

    # Return results
    result = {
        'credit_score': prediction[0][0],
        'debt_to_income_ratio': prediction[0][1],
        'requested_loan_amount': prediction[0][2],
        'approved_loan_amount': prediction[0][3],
        'is_creditworthy': bool(is_creditworthy[0])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
