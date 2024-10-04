import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

class feature_engineering():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def aggregate_features(self):
        customer_aggregate = self.df.groupby('CustomerId').agg(
            Total_transaction_amount = ('Amount', 'sum'),
            Average_transaction_amount = ('Amount', 'mean'),
            Transaction_count = ('Amount', 'count'),
            Std_transaction_amount = ('Amount', 'std')
        ).reset_index()

        # sort the values in descending order to get the top 20
        top_customers = customer_aggregate.sort_values(by='Total_transaction_amount', ascending=False).head(20)
        display(top_customers)
        display(customer_aggregate.head())

        # Display the top 20 transactions
        sns.barplot(x='Total_transaction_amount', y='CustomerId', data=top_customers)
        plt.title('Top 20 Customers by Total Transaction Amount')
        plt.show()

    def extract_features(self):

        # Convert the TransactionStartTime column to datetime
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])

        # Extract hour, day, month, and year
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year

        display((self.df[['Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year']]).head())



