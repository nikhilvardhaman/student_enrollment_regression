import os
import pickle
import numpy as np
import mlflow.pyfunc
from flask import Flask, request, jsonify, render_template

# Load the scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load the Ridge model from MLflow artifacts
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Set your MLflow tracking URI
# model_uri = 'runs:/51723cd0268143dba6389d890746deba/model'  # Replace with correct model path
# ridge_model = mlflow.pyfunc.load_model(model_uri)

ridge_model = pickle.load(open("ridge_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input features from form
        features = [
            float(request.form['gre_score']),
            float(request.form['toefl_score']),
            int(request.form['university_rating']),
            float(request.form['sop']),
            float(request.form['lor']),
            float(request.form['cgpa']),
            int(request.form['research'])
        ]
        
        # Preprocess input
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction = ridge_model.predict(features_scaled)[0]

        return render_template("index.html", prediction_text=f"Predicted Chance of Admit: {prediction:.2f}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
