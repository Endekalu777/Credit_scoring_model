import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from rfm_calculation import FeatureEngineeringWoE

app = Flask(__name__)

# Load the scalers and PCA from pickle files
minmax_scaler = joblib.load('minmax_scaler.pkl')
std_scaler = joblib.load('std_scaler.pkl')
pca = joblib.load('pca.pkl')

@app.route('/')
def home():
    return render_template('input_form.html')

@app.route('/rfm', methods=['POST'])
def calculate_rfms():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']

        # Determine the file extension
        file_ext = os.path.splitext(file.filename)[1].lower()

        # Read the file into a DataFrame
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file)
            elif file_ext == '.xlsx':
                df = pd.read_excel(file)
            else:
                return jsonify({'error': 'Unsupported file format. Please upload a CSV or Excel file.'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        # Ensure necessary columns are present
        required_columns = ['CustomerId', 'TransactionId', 'Amount', 'TransactionStartTime', 'ProductCategory', 'CurrencyCode']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Missing required columns in the file.'}), 400

        # Perform feature engineering
        feature_engineering = FeatureEngineeringWoE(df)
        feature_engineering.aggregate_features()
        feature_engineering.extract_features()
        feature_engineering.label_encoding()
        
        # Normalize numerical feature for training
        df['Normalized_Amount'] = minmax_scaler.transform(df[['Amount']])
        df['Standardized_Amount'] = std_scaler.transform(df[['Amount']])

        # Calculate RFM data
        rfms_data = feature_engineering.calculate_rfms()

        # Perform WoE binning
        train_final, test_final = feature_engineering.perform_woe_binning(rfms_data)

        # Ensure that test_final is not empty
        if test_final.empty:
            return jsonify({'error': 'The test data frame is empty after WoE binning.'}), 400

        # Scale the features in test_final
        test_final['Normalized_Amount'] = minmax_scaler.transform(test_final[['Amount']])
        test_final['Standardized_Amount'] = std_scaler.transform(test_final[['Amount']])

        # Apply PCA to test_final
        features_for_pca_test = test_final[['Normalized_Amount', 'Standardized_Amount']]  # Modify based on required features
        pca_transformed_test = pca.transform(features_for_pca_test)

        # Create a DataFrame for PCA results
        pca_columns = [f'PC{i+1}' for i in range(pca_transformed_test.shape[1])]
        pca_df_test = pd.DataFrame(pca_transformed_test, columns=pca_columns)

        # Concatenate PCA results with test_final
        test_final = pd.concat([test_final.reset_index(drop=True), pca_df_test], axis=1)

        # Load and predict with the logistic regression model
        model_path = 'models/Logistic_Regression.pkl'
        logistic_regression_model = joblib.load(model_path)

        # Make predictions using the logistic regression model
        required_features = [
            "Recency",
            "Frequency",
            "Monetary",
            "Size",
            "Size_woe",
            "Monetary_woe",
            "Frequency_woe",
            "Recency_woe",
            # Add PCA components if needed
            # 'PC1', 'PC2', ... # Uncomment and adjust if PCA results are used in prediction
        ]
        
        test_features = test_final[required_features]  # Ensure only required columns are included
        logistic_regression_predictions = logistic_regression_model.predict(test_features).tolist()

        # Render a result template to display the rfms_data and predictions
        return render_template('result.html', rfms_data=rfms_data.to_dict(orient='records'), predictions=logistic_regression_predictions)

if __name__ == '__main__':
    app.run(debug=True)
