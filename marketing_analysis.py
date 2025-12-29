"""
Marketing Analytics - Base Analysis
Analyzes campaign performance, customer segmentation, and product insights
Author: Sakshi Nakashe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

print("\nMarketing Analytics - Base Analysis")
print("=" * 70)
print("\nLoading datasets...")

# Load all datasets with error handling
try:
    customers = pd.read_csv('customers.csv')
    campaigns = pd.read_csv('campaigns.csv')
    products = pd.read_csv('products.csv')
    transactions = pd.read_csv('transactions.csv')
    print(f"✓ Customers: {len(customers):,} records")
    print(f"✓ Campaigns: {len(campaigns):,} records")
    print(f"✓ Products: {len(products):,} records")
    print(f"✓ Transactions: {len(transactions):,} records")
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("Please ensure all CSV files are in the same directory as this script.")
    exit()

print("\n" + "=" * 70)
print("\nCleaning and preparing data...")

# Convert date columns to datetime
customers['signup_date'] = pd.to_datetime(customers['signup_date'])
campaigns['start_date'] = pd.to_datetime(campaigns['start_date'])
campaigns['end_date'] = pd.to_datetime(campaigns['end_date'])
products['launch_date'] = pd.to_datetime(products['launch_date'])
transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])

# Clean transactions data
transactions_clean = transactions.dropna(subset=['product_id'])
print(f"✓ Removed {len(transactions) - len(transactions_clean):,} transactions with missing product_id")

transactions_clean['product_id'] = transactions_clean['product_id'].astype(int)

# Calculate net revenue (accounting for refunds)
transactions_clean['net_revenue'] = transactions_clean.apply(
    lambda row: -row['gross_revenue'] if row['refund_flag'] == 1 else row['gross_revenue'],
    axis=1
)

# Extract date components for analysis
transactions_clean['date'] = transactions_clean['timestamp'].dt.date
transactions_clean['month'] = transactions_clean['timestamp'].dt.to_period('M')
transactions_clean['year'] = transactions_clean['timestamp'].dt.year
transactions_clean['day_of_week'] = transactions_clean['timestamp'].dt.day_name()
print(f"✓ Calculated net revenue and date components")

print("\n" + "=" * 70)
print("\nMerging datasets...")

# Merge all datasets to create master dataset
trans_cust = transactions_clean.merge(customers, on='customer_id', how='left')
trans_cust_camp = trans_cust.merge(campaigns, on='campaign_id', how='left')
trans_full = trans_cust_camp.merge(products, on='product_id', how='left')

print(f"✓ Master dataset created: {len(trans_full):,} records with {trans_full.shape[1]} columns")

print("\n" + "=" * 70)
print("\nAnalyzing campaign performance...")

# Campaign Performance Analysis
campaign_trans = trans_full[trans_full['campaign_id'].notna()].copy()

campaign_performance = campaign_trans.groupby('campaign_id').agg({
    'transaction_id': 'count',
    'customer_id': 'nunique',
    'net_revenue': 'sum',
    'discount_applied': 'mean',
    'refund_flag': 'sum'
}).reset_index()

campaign_performance.columns = [
    'campaign_id', 
    'total_transactions', 
    'unique_customers', 
    'total_revenue', 
    'avg_discount',
    'total_refunds'
]

# Calculate additional campaign metrics
campaign_performance['avg_revenue_per_transaction'] = (
    campaign_performance['total_revenue'] / campaign_performance['total_transactions']
)

campaign_performance['refund_rate'] = (
    campaign_performance['total_refunds'] / campaign_performance['total_transactions'] * 100
)

# Merge with campaign details
campaign_performance = campaign_performance.merge(
    campaigns[['campaign_id', 'channel', 'objective', 'expected_uplift']], 
    on='campaign_id'
)

campaign_performance = campaign_performance.sort_values('total_revenue', ascending=False)

print(f"\nTop 5 Campaigns by Revenue:")
print(campaign_performance.head()[['campaign_id', 'channel', 'total_revenue', 'unique_customers']])

print("\n" + "=" * 70)
print("\nAnalyzing channels...")

# Acquisition Channel Analysis
acquisition_performance = trans_full.groupby('acquisition_channel').agg({
    'customer_id': 'nunique',
    'transaction_id': 'count',
    'net_revenue': 'sum',
    'gross_revenue': 'sum'
}).reset_index()

acquisition_performance.columns = [
    'acquisition_channel',
    'unique_customers',
    'total_transactions',
    'net_revenue',
    'gross_revenue'
]

acquisition_performance['transactions_per_customer'] = (
    acquisition_performance['total_transactions'] / acquisition_performance['unique_customers']
)

acquisition_performance['revenue_per_customer'] = (
    acquisition_performance['net_revenue'] / acquisition_performance['unique_customers']
)

acquisition_performance = acquisition_performance.sort_values('net_revenue', ascending=False)

print(f"\nAcquisition Channel Performance:")
print(acquisition_performance[['acquisition_channel', 'net_revenue', 'unique_customers', 'revenue_per_customer']])

# Campaign Channel Analysis
campaign_channel_perf = campaign_trans.groupby('channel').agg({
    'transaction_id': 'count',
    'net_revenue': 'sum',
    'customer_id': 'nunique'
}).reset_index()

campaign_channel_perf.columns = ['channel', 'transactions', 'revenue', 'customers']
campaign_channel_perf = campaign_channel_perf.sort_values('revenue', ascending=False)

print(f"\nCampaign Channel Performance:")
print(campaign_channel_perf)

print("\n" + "=" * 70)
print("\nPerforming RFM customer segmentation...")

# RFM Analysis
max_date = transactions_clean['timestamp'].max()

rfm = transactions_clean.groupby('customer_id').agg({
    'timestamp': lambda x: (max_date - x.max()).days,
    'transaction_id': 'count',
    'net_revenue': 'sum'
}).reset_index()

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Create RFM scores (1-5 scale)
rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')

# Combined RFM score
rfm['rfm_score'] = (
    rfm['r_score'].astype(str) + 
    rfm['f_score'].astype(str) + 
    rfm['m_score'].astype(str)
)

rfm['rfm_score_avg'] = (
    rfm['r_score'].astype(int) + 
    rfm['f_score'].astype(int) + 
    rfm['m_score'].astype(int)
) / 3

# Segment customers based on RFM scores
def segment_customers(row):
    r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
    
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Lost'
    else:
        return 'Potential'

rfm['segment'] = rfm.apply(segment_customers, axis=1)

print(f"\nCustomer Segment Distribution:")
print(rfm['segment'].value_counts())

segment_summary = rfm.groupby('segment').agg({
    'customer_id': 'count',
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'
}).round(2)

segment_summary.columns = ['customer_count', 'avg_recency_days', 'avg_frequency', 'avg_monetary']

print(f"\nSegment Characteristics:")
print(segment_summary)

print("\n" + "=" * 70)
print("\nAnalyzing product performance...")

# Product Performance Analysis
product_performance = trans_full.groupby('product_id').agg({
    'transaction_id': 'count',
    'net_revenue': 'sum',
    'customer_id': 'nunique',
    'category': 'first',
    'brand': 'first',
    'is_premium': 'first'
}).reset_index()

product_performance.columns = [
    'product_id', 'units_sold', 'total_revenue', 
    'unique_customers', 'category', 'brand', 'is_premium'
]

product_performance = product_performance.sort_values('total_revenue', ascending=False)

print(f"\nTop 10 Products by Revenue:")
print(product_performance.head(10)[['product_id', 'category', 'brand', 'total_revenue', 'units_sold']])

# Category Performance
category_performance = trans_full.groupby('category').agg({
    'transaction_id': 'count',
    'net_revenue': 'sum',
    'customer_id': 'nunique',
    'discount_applied': 'mean'
}).reset_index()

category_performance.columns = [
    'category', 'transactions', 'revenue', 
    'customers', 'avg_discount'
]

category_performance = category_performance.sort_values('revenue', ascending=False)

print(f"\nCategory Performance:")
print(category_performance)

# Premium vs Regular Analysis
premium_analysis = trans_full.groupby('is_premium').agg({
    'transaction_id': 'count',
    'net_revenue': 'sum',
    'discount_applied': 'mean'
}).reset_index()

premium_analysis['is_premium'] = premium_analysis['is_premium'].map({1: 'Premium', 0: 'Regular'})
premium_analysis.columns = ['product_type', 'transactions', 'revenue', 'avg_discount']

print(f"\nPremium vs Regular Products:")
print(premium_analysis)

print("\n" + "=" * 70)
print("\nKey Metrics Summary:")

# Overall Business Metrics
total_revenue = trans_full['net_revenue'].sum()
total_transactions = len(trans_full)
total_customers = trans_full['customer_id'].nunique()
avg_order_value = total_revenue / total_transactions
avg_discount = trans_full['discount_applied'].mean()

print(f"\nTotal Revenue: ${total_revenue:,.2f}")
print(f"Total Transactions: {total_transactions:,}")
print(f"Total Customers: {total_customers:,}")
print(f"Average Order Value: ${avg_order_value:.2f}")
print(f"Average Discount: {avg_discount:.2%}")

# Campaign Metrics
campaign_revenue = campaign_trans['net_revenue'].sum()
campaign_contribution = (campaign_revenue / total_revenue) * 100

print(f"\nCampaign-Attributed Revenue: ${campaign_revenue:,.2f}")
print(f"Campaign Contribution: {campaign_contribution:.1f}%")

# Customer Lifetime Value by Segment
ltv_by_segment = rfm.groupby('segment')['monetary'].mean().sort_values(ascending=False)

print(f"\nAverage Customer Lifetime Value by Segment:")
for segment, value in ltv_by_segment.items():
    print(f"  {segment}: ${value:,.2f}")

print("\n" + "=" * 70)
print("\nSaving analysis results...")

# Save all analysis outputs
campaign_performance.to_csv('analysis_campaign_performance.csv', index=False)
acquisition_performance.to_csv('analysis_acquisition_channels.csv', index=False)
rfm.to_csv('analysis_customer_segments.csv', index=False)
product_performance.to_csv('analysis_product_performance.csv', index=False)
category_performance.to_csv('analysis_category_performance.csv', index=False)
trans_full.to_csv('master_dataset_for_tableau.csv', index=False)

print(f"\n✓ analysis_campaign_performance.csv")
print(f"✓ analysis_acquisition_channels.csv")
print(f"✓ analysis_customer_segments.csv")
print(f"✓ analysis_product_performance.csv")
print(f"✓ analysis_category_performance.csv")
print(f"✓ master_dataset_for_tableau.csv")

print("\n" + "=" * 70)
print("\n✅ Base analysis complete!")
print("Next: Run advanced_analytics.py for churn prediction and attribution analysis\n")