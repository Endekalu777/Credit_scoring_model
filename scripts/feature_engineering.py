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




