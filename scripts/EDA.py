import pandas as pd
from IPython.display import display
from scipy.stats import stats



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

    





