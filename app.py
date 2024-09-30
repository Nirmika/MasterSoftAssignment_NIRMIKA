from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('models/best_model.pkl')
encoders = joblib.load('models/encoders.pkl')
scaler = joblib.load('models/scaler.pkl')
numeral_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlyCharge']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data
        input_data = request.json
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data], columns=input_data.keys())
        # --- Feature Engineering ---
        # Add 'StreamingServices' if it doesn't already exist
        if 'StreamingServices' not in input_df.columns:
            input_df['StreamingServices'] = (input_df['StreamingTV'] == 1) & (input_df['StreamingMovies'] == 1)
            input_df['StreamingServices'] = input_df['StreamingServices'].astype(int)
        
        # Add 'HasOnlineSecurityAndProtection' if it doesn't already exist
        if 'HasOnlineSecurityAndProtection' not in input_df.columns:
            input_df['HasOnlineSecurityAndProtection'] = (input_df['OnlineSecurity'] == 1) & (input_df['OnlineBackup'] == 1)
            input_df['HasOnlineSecurityAndProtection'] = input_df['HasOnlineSecurityAndProtection'].astype(int)
        
        # Create 'AverageMonthlyCharge' feature
        input_df['AverageMonthlyCharge'] = input_df['TotalCharges'] / input_df['tenure']

        # Replace infinite values with NaN and then fill NaN with median
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df['AverageMonthlyCharge'].fillna(input_df['AverageMonthlyCharge'].median(), inplace=True)

        # Check and encode categorical columns dynamically
        for col in input_df.columns:
            if col in encoders:
                
                input_df[col] = encoders[col].transform(input_df[col])

        # Scale numerical columns
        input_df[numeral_cols] = scaler.transform(input_df[numeral_cols])
        # Make prediction using the model
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            result = "Customer will churn"
        else:
            result = "Customer will not churn"
        
        return jsonify({
            'prediction': result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def welcome():
    return "Welcome to the Customer Churn Prediction"

if __name__ == '__main__':
    app.run(debug=True)
