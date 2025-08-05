from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("loan_approval_model.pkl")

# Mapping dictionaries
job_map = {
    0: "Admin", 1: "Blue-collar", 2: "Entrepreneur", 3: "Housemaid",
    4: "Management", 5: "Retired", 6: "Self-employed", 7: "Services",
    8: "Student", 9: "Technician", 10: "Unemployed", 11: "Unknown"
}

marital_map = {0: "Divorced", 1: "Married", 2: "Single"}
education_map = {0: "Primary", 1: "Secondary", 2: "Tertiary", 3: "Unknown"}
housing_map = {0: "No", 1: "Yes"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get raw inputs (as strings)
        age = int(request.form['age'])
        job = int(request.form['job'])
        marital = int(request.form['marital'])
        education = int(request.form['education'])
        balance = int(request.form['balance'])
        housing = int(request.form['housing'])
        duration = int(request.form['duration'])
        campaign = int(request.form['campaign'])

        # For prediction
        features = np.array([[age, job, marital, education, balance, housing, duration, campaign]])
        prediction = model.predict(features)
        result = "Yes" if prediction[0] == 1 else "No"

        # Decode values back for displaying
        decoded_input = {
            "Age": age,
            "Job": job_map[job],
            "Marital Status": marital_map[marital],
            "Education": education_map[education],
            "Balance": balance,
            "Housing Loan": housing_map[housing],
            "Duration": duration,
            "Campaign": campaign
        }

        return render_template('index.html', result=result, decoded_input=decoded_input)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
