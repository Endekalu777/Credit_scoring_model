import pandas as pd
from IPython.display import display
from scipy.stats import stats
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



class EDA():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.numerical_cols = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']
        self.numerical_data = self.df[self.numerical_cols]

    def overview(self):
        print(f"Shape of the dataset: {self.df.shape}\n")
        print("Preview of the first 5 rows:")
        display(self.df.head())
        print("Data types of the dataset")
        display(self.df.dtypes)
        print("Missing values of the dataset: ")
        display(self.df.isnull().sum())

    def distribution(self):
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

    def visualize_distribution(self):
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
        # List of categorical columns to analyze
        categorical_cols = [
            'CurrencyCode', 'ProviderId', 'ProductCategory', 'ChannelId'
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
        corr_matrix = self.df[self.numerical_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)

        # Set title and labels
        plt.title('Correlation Matrix of Numerical Features', fontsize=16)
        plt.show()


        





