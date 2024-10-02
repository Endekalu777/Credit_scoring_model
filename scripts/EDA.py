import pandas as pd
from IPython.display import display
from scipy.stats import stats



class EDA():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def overview(self):
        print(f"Shape of the dataset: {self.df.shape}\n")
        print("Preview of the first 5 rows:")
        display(self.df.head())
        print("Data types of the dataset")
        display(self.df.dtypes)
    


