import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



class EDA():
    def __init__(self, filepath):
        """
        Initialize the EDA class with a CSV filepath.
        Parameters:
            filepath (str): The file path to the dataset.
        """
        self.df = pd.read_csv(filepath)
        self.numerical_cols = ['Amount', 'Value']
        self.numerical_data = self.df[self.numerical_cols]

    def overview(self):
        """
        Provide an overview of the dataset structure, including shape, preview, 
        data types, and missing values.
        """
        print(f"Shape of the dataset: {self.df.shape}\n")
        print("Preview of the first 5 rows:")
        display(self.df.head())
        print("Data types of the dataset")
        display(self.df.dtypes)
        print("Missing values of the dataset: ")
        display(self.df.isnull().sum())

    def distribution(self):
        """
        Display statistical summary and calculate skewness and kurtosis
        for numerical columns to understand distribution shape.
        """
        print("Statistical summary of the dataset\n")
        display(self.df.describe())

        # Understand shape of distribution
        shape = {
                'Skewness': self.numerical_data.apply(stats.skew),
                'Kurtosis': self.numerical_data.apply(stats.kurtosis)
                }
        
        # Print each column's skewness and kurtosis with proper formatting
        print("\nShape of distribution:")
        for metric, values in shape.items():
            print(f"\n{metric}:")
            for col, val in values.items():
                print(f"{col}: {val}")

    def handle_negative_values(self):
        """
        Identify and visualize negative values in the 'Amount' column, 
        along with fraud status for negative transactions.
        """
        # Filter negative transactions
        negative_transactions= self.df[self.df['Amount'] < 0]
        print(f"Number of negative values: {len(negative_transactions)}")

        # Distribution of the negative values
        sns.histplot(negative_transactions['Amount'], kde=True)
        plt.title('Distribution of Negative Amount Values')
        plt.show()

        # Fraud status for negative transactions
        sns.countplot(x='FraudResult', data=negative_transactions)
        plt.title('Fraud Status for Negative Transactions')
        plt.show()

    def visualize_distribution(self):
        """
        Visualize the distribution of numerical features using histograms, 
        boxplots, and density plots to detect patterns and potential outliers.
        """
        # Plot histograms for each numerical feature
        self.numerical_data.hist(bins=30, figsize=(10, 8), layout=(3, 2))
        plt.tight_layout()
        plt.show()

        # Plot boxplots for each numerical feature
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(self.numerical_cols, 1):
            plt.subplot(3, 2, i)
            sns.boxplot(x=self.numerical_data[col])
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

        # Plot density plots for each numerical feature
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(self.numerical_cols, 1):
            plt.subplot(3, 2, i)
            sns.kdeplot(self.numerical_data[col], shade=True)
            plt.title(f'Density Plot of {col}')
        plt.tight_layout()
        plt.show()
    
    def categorical_distribution(self):
        """
        Analyze the distribution of categorical features using bar plots 
        to understand the frequency of categories.
        """
        # List of categorical columns to analyze
        categorical_cols = [
            'CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId',
            'PricingStrategy', 'FraudResult'
        ]

        # Set up the matplotlib figure
        plt.figure(figsize=(15, 20))

        # Loop through the categorical columns and create a bar plot for each
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(len(categorical_cols), 1, i)
            sns.countplot(y=col, data=self.df, order=self.df[col].value_counts().index)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)

        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """
        Perform correlation analysis on numerical features and visualize the 
        correlation matrix using a heatmap to understand relationships.
        """
        self.numerical_cols = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']
        corr_matrix = self.df[self.numerical_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)

        # Set title and labels
        plt.title('Correlation Matrix of Numerical Features', fontsize=16)
        plt.show()

    def log_transformation(self):
        """
        Apply log transformation to 'Amount' and 'Value' columns and 
        visualize the effect of the transformation.
        """
        # Apply log transformation
        self.df['Log_Amount'] = np.log1p(self.df['Amount'])
        self.df['Log_Value'] = np.log1p(self.df['Value'])

        # Visualize the transformation
        plt.figure(figsize=(12, 5))

        plt.subplot(121)
        sns.histplot(self.df['Amount'], kde=True)
        plt.title('Original Amount Distribution')
        plt.xlabel('Amount')

        plt.subplot(122)
        sns.histplot(self.df['Log_Amount'], kde=True)
        plt.title('Log-Transformed Amount Distribution')
        plt.xlabel('Log(Amount + 1)')

        plt.tight_layout()
        plt.show()


        





