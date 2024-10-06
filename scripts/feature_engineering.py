import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import scorecardpy as sc
from monotonic_binning.monotonic_woe_binning import Binning
from datetime import datetime
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

class feature_engineering_WoE():
    # Initialize the feature_engineering class by loading the dataset.   
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def aggregate_features(self):
        """
        Aggregate customer transaction data by CustomerId, calculating
        total, average, count, and standard deviation of transaction amounts.
        Display the top 20 customers based on total transaction amounts.
        """
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
        """
        Extract additional features from the TransactionStartTime column,
        including transaction hour, day, month, and year.
        """

        # Convert the TransactionStartTime column to datetime
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])

        # Extract hour, day, month, and year
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year

        display((self.df[['Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year']]).head())

    def label_encoding(self):
        """
        Perform label encoding on categorical features: ProductCategory and CurrencyCode.
        Display a sample of encoded columns and provide mappings of labels to encoded values.
        """
        # Initialize the LabelEncoder
        le_product = LabelEncoder()
        le_currency = LabelEncoder()

        # Apply label encoding on ProductCategory
        self.df['ProductCategory_encoded'] = le_product.fit_transform(self.df['ProductCategory'])

        # Apply label encoding on CurrencyCode
        self.df['CurrencyCode_encoded'] = le_currency.fit_transform(self.df['CurrencyCode'])

        # Display a sample of the DataFrame with original and encoded columns (first 5 rows)
        display(self.df[['ProductCategory', 'ProductCategory_encoded', 'CurrencyCode', 'CurrencyCode_encoded']].head())

        # Create and display the mapping of labels to encoded values using a sample of encoded values
        product_mapping = pd.DataFrame({
            'Original': le_product.inverse_transform(range(len(le_product.classes_))),
            'Encoded': range(len(le_product.classes_))
        })

        currency_mapping = pd.DataFrame({
            'Original': le_currency.inverse_transform(range(len(le_currency.classes_))),
            'Encoded': range(len(le_currency.classes_))
        })

        # Display the mappings for ProductCategory and CurrencyCode (first 5 rows)
        display(product_mapping.head())
        display(currency_mapping.head())

    def normalize_numerical_feature(self):
        """
        Normalize and standardize the 'Amount' column using MinMaxScaler and StandardScaler.
        Display a sample of the original, normalized, and standardized values.
        """
        # Normalize and standardize numerical features
        scaler = MinMaxScaler()
        self.df['Normalized_Amount'] = scaler.fit_transform(self.df[['Amount']])
        # Normalize and standardize numerical features
        scaler = StandardScaler()
        self.df['Standardized_Amount'] = scaler.fit_transform(self.df[['Amount']])
        normalized_cols = ['Amount', 'Normalized_Amount', 'Standardized_Amount']
        display(self.df[normalized_cols].head())

    def calculate_rfms(self):
        # Calculate Recency, Frequency, Monetary, and Size (RFMS) metrics
        latest_date = self.df['TransactionStartTime'].max()
        rfms = self.df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (latest_date - x.max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Amount': ['sum', 'mean']  # Monetary and Size
        }).reset_index()
        
        rfms.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Size']
        
        # Normalize RFMS metrics
        scaler = MinMaxScaler()
        rfms[['Recency', 'Frequency', 'Monetary', 'Size']] = scaler.fit_transform(rfms[['Recency', 'Frequency', 'Monetary', 'Size']])
        
        # Calculate RFMS score
        rfms['RFMS_Score'] = rfms[['Recency', 'Frequency', 'Monetary', 'Size']].sum(axis=1)
        
        # Classify customers into good or bad based on the median RFMS_Score
        median_score = rfms['RFMS_Score'].median()
        rfms['Label'] = (rfms['RFMS_Score'] >= median_score).astype(int)  # Good (1), Bad (0)
        
        # Visualization
        top_customers = rfms.sort_values(by='RFMS_Score', ascending=False).head(20)
        sns.barplot(x='RFMS_Score', y='CustomerId', data=top_customers)
        plt.title('Top 20 Customers by RFMS Score')
        plt.show()
        
        return rfms
    
    def perform_woe_binning(self, rfms):
        # Step 1: Split data into training and testing sets (70% train, 30% test)
        train, test = sc.split_df(rfms, y='Label', ratio=0.7, seed=999).values()

        # Step 2: Prepare features for WoE binning
        features_for_woe = ['Recency', 'Frequency', 'Monetary', 'Size', 'RFMS_Score']

        # Step 3: WoE binning using scorecardpy
        bins_adj = sc.woebin(train, y='Label', x=features_for_woe, positive='bad|0')

        # Step 4: Apply WoE binning to the training and test datasets
        train_woe = sc.woebin_ply(train, bins_adj)
        test_woe = sc.woebin_ply(test, bins_adj)

        # Step 5: WoE Visualization (WoE plots for each feature)
        for feature in features_for_woe:
            if feature in bins_adj:
                plt.figure(figsize=(12, 8))  # Adjusted figure size to make it clearer
                sc.woebin_plot(bins_adj[feature])

                # Customizing the plot for better aesthetics
                plt.title(f'WoE Plot for {feature}', fontsize=6)  # Larger title font
                plt.xlabel('Binned Values', fontsize=4)  # Larger x-axis label font
                plt.ylabel('WoE', fontsize=6)  # Larger y-axis label font
                plt.xticks(rotation=0, ha='right', fontsize=6)  # Rotate x-ticks for readability
                plt.yticks(fontsize=5)  # Adjust y-tick font size
                plt.grid(True, linestyle='--', alpha=0.7)  # Light grid lines for readability
                plt.tight_layout(pad=3.0)  # Ensure no overlapping of elements
                plt.show()
                plt.close()

        # Step 6: Merging the transformed WoE values back into the original datasets
        train_final = train.merge(train_woe, how='left', left_index=True, right_index=True)
        test_final = test.merge(test_woe, how='left', left_index=True, right_index=True)

        # Remove duplicates and rename columns
        train_final = train_final.drop(['CustomerId_y', 'Label_y'], axis=1)
        test_final = test_final.drop(['CustomerId_y', 'Label_y'], axis=1)
        train_final = train_final.rename(columns={'CustomerId_x': 'CustomerId', 'Label_x': 'Label'})
        test_final = test_final.rename(columns={'CustomerId_x': 'CustomerId', 'Label_x': 'Label'})

        return train_final, test_final