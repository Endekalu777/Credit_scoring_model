# app/model_training.py

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib 

class ModelSelectionAndTraining:
    def __init__(self, train_final_path, test_final_path):
        # Load the dataset
        train_final = pd.read_csv(train_final_path)
        test_final = pd.read_csv(test_final_path)
        
        # Split into features and labels
        self.X_train = train_final.drop(columns=['CustomerId', 'Label', 'RFMS_Score', 'RFMS_Score_woe'])
        self.y_train = train_final['Label']
        self.X_test = test_final.drop(columns=['CustomerId', 'Label', 'RFMS_Score', 'RFMS_Score_woe'])
        self.y_test = test_final['Label']
        
        # Initialize scaler and PCA
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

    def model_selection_and_training(self):
        # Fit the scaler and PCA on the training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)

        # Define pipelines for each model without scaling and PCA steps
        pipelines = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', C=0.001, penalty='l2', solver='liblinear'),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=2, min_samples_leaf=10),
            'Random Forest': RandomForestClassifier(class_weight='balanced', max_depth=2, n_estimators=10, min_samples_leaf=10)
        }

        # Set the MLflow tracking URI to a custom folder path
        mlflow.set_tracking_uri("../mlruns")

        # Initialize MLflow
        mlflow.set_experiment("Model Selection Experiment")

        # Prepare to plot ROC curves
        plt.figure(figsize=(10, 8))

        for name, model in pipelines.items():
            with mlflow.start_run(run_name=name):
                print(f'Training {name}...')

                # Train the model on PCA-transformed data
                model.fit(self.X_train_pca, self.y_train)

                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train_pca, self.y_train, cv=5, scoring='accuracy')
                mlflow.log_metric('cv_accuracy', cv_scores.mean())
                print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

                # Predictions and metrics calculation
                y_pred = model.predict(self.X_test_pca)
                y_pred_proba = model.predict_proba(self.X_test_pca)[:, 1]  # Get probabilities for ROC-AUC
                
                # Metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, zero_division=0)
                recall = recall_score(self.y_test, y_pred, zero_division=0)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)

                # Log metrics to MLflow
                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_metric('precision', precision)
                mlflow.log_metric('recall', recall)
                mlflow.log_metric('f1_score', f1)
                mlflow.log_metric('roc_auc', roc_auc)

                # Save the trained model as a PKL file
                joblib.dump(model, f'../models/{name.replace(" ", "_")}.pkl')
                print(f'Model {name} saved as PKL file.')

                # Print metrics
                print(f'Model: {name}')
                print(f'Accuracy: {accuracy:.4f}')
                print(f'Precision: {precision:.4f}')
                print(f'Recall: {recall:.4f}')
                print(f'F1 Score: {f1:.4f}')
                print(f'ROC AUC: {roc_auc:.4f}')
                print('-' * 30)

                # Plot ROC Curve
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

        # Plot formatting for ROC curves
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')  # Diagonal line for random guessing
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        # Save the scaler and PCA models
        joblib.dump(self.scaler, '../models/global_scaler.pkl')
        joblib.dump(self.pca, '../models/global_pca.pkl')
