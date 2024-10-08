import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from rfm_calculation import FeatureEngineeringWoE

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('input_form.html')

@app.route('/rfm', methods=['POST'])
def calculate_rfms():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_excel(file)
        
        feature_engineering = FeatureEngineeringWoE(df)
        rfms_data, raw_rfms_data = feature_engineering.calculate_rfms()
        final_data = feature_engineering.perform_woe_binning(rfms_data)

        model_path = 'models/Decision_Tree.pkl'
        decision_tree_model = joblib.load(model_path)

        required_features = [
            "Recency", "Frequency", "Monetary", "Size",
            "Size_woe", "Monetary_woe", "Frequency_woe", "Recency_woe"
        ]
        
        prediction_features = final_data[required_features]
        predictions = decision_tree_model.predict(prediction_features)
        probabilities = decision_tree_model.predict_proba(prediction_features)[:, 1]

        # Create a separate prediction table
        prediction_table = pd.DataFrame({
            'CustomerId': final_data['CustomerId'],
            'Prediction': predictions,
            'Probability': probabilities
        })

        return render_template('result.html', 
                               result_data=rfms_data.to_dict(orient='records'),
                               raw_rfms_data=raw_rfms_data.to_dict(orient='records'),
                               prediction_data=prediction_table.to_dict(orient='records'),
                               transaction_data=df.to_dict(orient='records'))
if __name__ == '__main__':
    app.run(debug=True)