# app/rfm_calculation.py

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
        latest_date = self.df['TransactionStartTime'].max()
        rfms = self.df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (latest_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': ['sum', 'mean']
        }).reset_index()

        rfms.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Size']
        scaler = MinMaxScaler()
        rfms[['Recency', 'Frequency', 'Monetary', 'Size']] = scaler.fit_transform(rfms[['Recency', 'Frequency', 'Monetary', 'Size']])
        rfms['RFMS_Score'] = rfms[['Recency', 'Frequency', 'Monetary', 'Size']].sum(axis=1)
        median_score = rfms['RFMS_Score'].median()
        rfms['Label'] = (rfms['RFMS_Score'] >= median_score).astype(int)

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

        # Step 6: Merging the transformed WoE values back into the original datasets
        train_final = train.merge(train_woe, how='left', left_index=True, right_index=True)
        test_final = test.merge(test_woe, how='left', left_index=True, right_index=True)

        # Remove duplicates and rename columns
        train_final = train_final.drop(['CustomerId_y', 'Label_y'], axis=1)
        test_final = test_final.drop(['CustomerId_y', 'Label_y'], axis=1)
        train_final = train_final.rename(columns={'CustomerId_x': 'CustomerId', 'Label_x': 'Label'})
        test_final = test_final.rename(columns={'CustomerId_x': 'CustomerId', 'Label_x': 'Label', 'Frequency_x': 'Frequency', 'Frequency_y': 'Frequency_woe'})

        return train_final, test_final