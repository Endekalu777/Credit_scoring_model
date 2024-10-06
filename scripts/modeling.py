import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ModelSelectionAndTraining:
    def __init__(self, train_final_path, test_final_path):
        # Load the dataset
        train_final = pd.read_csv(train_final_path)
        test_final = pd.read_csv(test_final_path)
        
        # Split into features and labels
        self.X_train = train_final.drop(columns=['CustomerId', 'Label'])
        self.y_train = train_final['Label']
        self.X_test = test_final.drop(columns=['CustomerId', 'Label'])
        self.y_test = test_final['Label']

        # Feature Scaling
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # Optional: Reduce dimensionality with PCA
        pca = PCA(n_components=0.95)  # Keep 95% of the variance
        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

    def model_selection_and_training(self):
        # Define models with increased regularization and additional parameters for simplicity
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', C=0.001, penalty='l2', solver='liblinear'),
            'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=2, min_samples_leaf=10),
            'Random Forest': RandomForestClassifier(class_weight='balanced', max_depth=2, n_estimators=10, min_samples_leaf=10)
        }

        # Step 3: Train and evaluate each model
        plt.figure(figsize=(10, 8))
        for name, model in models.items():
            print(f'Training {name}...')
            # Fit the model on the training data
            model.fit(self.X_train, self.y_train)

            # Step 4: Model evaluation with cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

            # Step 5: Predictions and metrics calculation
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get probabilities for ROC-AUC

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Print evaluation metrics
            print(f'Model: {name}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'ROC AUC: {roc_auc:.4f}')
            print('-' * 30)

            # Step 6: Plot ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

        # Plot formatting
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')  # Diagonal line for random guessing
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

