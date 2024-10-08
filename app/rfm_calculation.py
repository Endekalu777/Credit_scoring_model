import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import scorecardpy as sc
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureEngineeringWoE:
    def __init__(self, df):
        self.df = df

    def aggregate_features(self):
        customer_aggregate = self.df.groupby('CustomerId').agg(
            Total_transaction_amount=('Amount', 'sum'),
            Average_transaction_amount=('Amount', 'mean'),
            Transaction_count=('Amount', 'count'),
            Std_transaction_amount=('Amount', 'std')
        ).reset_index()

    def extract_features(self):
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year

    def label_encoding(self):
        le_product = LabelEncoder()
        le_currency = LabelEncoder()

        self.df['ProductCategory_encoded'] = le_product.fit_transform(self.df['ProductCategory'])
        self.df['CurrencyCode_encoded'] = le_currency.fit_transform(self.df['CurrencyCode'])

    def normalize_numerical_feature(self):
        scaler = MinMaxScaler()
        self.df['Normalized_Amount'] = scaler.fit_transform(self.df[['Amount']])
        scaler = StandardScaler()
        self.df['Standardized_Amount'] = scaler.fit_transform(self.df[['Amount']])

    def calculate_rfms(self):
        latest_date = pd.to_datetime(self.df['TransactionStartTime']).max()
        rfms = self.df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (latest_date - pd.to_datetime(x).max()).days,  # Recency
            'TransactionId': 'count',  # Frequency
            'Amount': ['sum', 'mean']  # Monetary (sum) and Size (mean)
        }).reset_index()

        # Rename columns
        rfms.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Size']
        
        # Store raw values (for reference if needed)
        raw_rfms = rfms.copy()
        
        # Normalize only the 'Recency' column
        min_val = rfms['Recency'].min()
        max_val = rfms['Recency'].max()
        if min_val == max_val:
            rfms['Recency'] = 1  # If all values are the same, set them to 1
        else:
            rfms['Recency'] = (rfms['Recency'] - min_val) / (max_val - min_val)
        
        # Invert Recency so that lower values (more recent) get higher scores
        rfms['Recency'] = 1 - rfms['Recency']
        
        # Calculate the RFMS score using raw 'Frequency', 'Monetary', 'Size' values
        rfms['RFMS_Score'] = rfms[['Recency', 'Frequency', 'Monetary', 'Size']].sum(axis=1)
        
        # Median score to label customers (binary classification)
        median_score = rfms['RFMS_Score'].median()
        rfms['Label'] = (rfms['RFMS_Score'] >= median_score).astype(int)

        return rfms, raw_rfms

    def perform_woe_binning(self, rfms):
        # Prepare features for WoE binning
        features_for_woe = ['Recency', 'Frequency', 'Monetary', 'Size', 'RFMS_Score']

        # WoE binning using scorecardpy
        bins_adj = sc.woebin(rfms, y='Label', x=features_for_woe, positive='bad|0')

        # Apply WoE binning to the dataset
        rfms_woe = sc.woebin_ply(rfms, bins_adj)

        # Merging the transformed WoE values back into the original dataset
        final_data = rfms.merge(rfms_woe, how='left', left_index=True, right_index=True)

        # Remove duplicates and rename columns
        final_data = final_data.drop(['CustomerId_y', 'Label_y'], axis=1)
        final_data = final_data.rename(columns={'CustomerId_x': 'CustomerId', 'Label_x': 'Label', 'Frequency_x': 'Frequency', 'Frequency_y': 'Frequency_woe'})

        return final_data