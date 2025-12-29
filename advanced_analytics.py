"""
Advanced Marketing Analytics
Multi-touch attribution, churn prediction, and budget optimization
Author: Sakshi Nakashe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("\nAdvanced Marketing Analytics")
print("=" * 70)
print("\nLoading previous analysis results...")

# Load results from base analysis
try:
    trans_full = pd.read_csv('master_dataset_for_tableau.csv')
    rfm = pd.read_csv('analysis_customer_segments.csv')
    campaigns_perf = pd.read_csv('analysis_campaign_performance.csv')
    print(f"✓ Master dataset: {len(trans_full):,} records")
    print(f"✓ Customer segments: {len(rfm):,} customers")
    print(f"✓ Campaign performance: {len(campaigns_perf)} campaigns")
except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("Please run marketing_analysis.py first to generate required files.")
    exit()

print("\n" + "=" * 70)
print("\nAnalyzing multi-touch attribution...")

# Multi-Touch Attribution Analysis
trans_full['timestamp'] = pd.to_datetime(trans_full['timestamp'])

# Count unique channels each customer engaged with
customer_touchpoints = trans_full.groupby('customer_id')['channel'].nunique().reset_index()
customer_touchpoints.columns = ['customer_id', 'num_channels']

# Get total revenue per customer
customer_revenue = trans_full.groupby('customer_id')['net_revenue'].sum().reset_index()

# Merge touchpoints with revenue
touch_analysis = customer_touchpoints.merge(customer_revenue, on='customer_id')
touch_analysis['is_multi_touch'] = (touch_analysis['num_channels'] > 1).astype(int)

# Calculate metrics for multi-touch vs single-touch
multi_touch = touch_analysis[touch_analysis['is_multi_touch'] == 1]
single_touch = touch_analysis[touch_analysis['is_multi_touch'] == 0]

multi_touch_count = len(multi_touch)
single_touch_count = len(single_touch)
multi_touch_pct = (multi_touch_count / len(touch_analysis)) * 100

multi_touch_avg_revenue = multi_touch['net_revenue'].mean()
single_touch_avg_revenue = single_touch['net_revenue'].mean()
uplift = ((multi_touch_avg_revenue / single_touch_avg_revenue) - 1) * 100

print(f"\nMulti-Touch Attribution Results:")
print(f"Total customers analyzed: {len(touch_analysis):,}")
print(f"Multi-touch customers: {multi_touch_count:,} ({multi_touch_pct:.1f}%)")
print(f"Single-touch customers: {single_touch_count:,} ({100-multi_touch_pct:.1f}%)")
print(f"\nRevenue Comparison:")
print(f"  Multi-touch average: ${multi_touch_avg_revenue:.2f}")
print(f"  Single-touch average: ${single_touch_avg_revenue:.2f}")
print(f"  Revenue uplift: {uplift:+.1f}%")

if len(multi_touch) > 0:
    avg_channels = multi_touch['num_channels'].mean()
    print(f"  Average channels engaged (multi-touch): {avg_channels:.1f}")

# First-touch and Last-touch Attribution
first_last = trans_full.sort_values(['customer_id', 'timestamp']).groupby('customer_id').agg({
    'channel': ['first', 'last'],
    'net_revenue': 'sum'
})
first_last.columns = ['first_channel', 'last_channel', 'revenue']
first_last = first_last.reset_index()

print(f"\nFirst-Touch Attribution (Customer Acquisition):")
first_touch_attr = first_last.groupby('first_channel')['revenue'].sum().sort_values(ascending=False)
for channel, rev in first_touch_attr.head().items():
    print(f"  {channel}: ${rev:,.2f}")

print(f"\nLast-Touch Attribution (Final Conversion):")
last_touch_attr = first_last.groupby('last_channel')['revenue'].sum().sort_values(ascending=False)
for channel, rev in last_touch_attr.head().items():
    print(f"  {channel}: ${rev:,.2f}")

# Save attribution analysis
touch_analysis.to_csv('analysis_attribution_fixed.csv', index=False)
print(f"\n✓ Saved: analysis_attribution_fixed.csv")

print("\n" + "=" * 70)
print("\nBuilding churn prediction model...")

# Churn Prediction Model
max_date = pd.to_datetime(trans_full['timestamp']).max()

customer_last_purchase = trans_full.groupby('customer_id').agg({
    'timestamp': lambda x: pd.to_datetime(x).max(),
    'transaction_id': 'count',
    'net_revenue': 'sum'
}).reset_index()

customer_last_purchase.columns = ['customer_id', 'last_purchase_date', 'total_transactions', 'total_revenue']
customer_last_purchase['days_since_purchase'] = (
    max_date - pd.to_datetime(customer_last_purchase['last_purchase_date'])
).dt.days

# Define churn threshold (75th percentile of inactivity)
churn_threshold = customer_last_purchase['days_since_purchase'].quantile(0.75)
customer_last_purchase['churned'] = (
    customer_last_purchase['days_since_purchase'] > churn_threshold
).astype(int)

print(f"\nChurn Definition:")
print(f"  Churn threshold: {churn_threshold:.0f} days of inactivity")
print(f"  Churned customers: {customer_last_purchase['churned'].sum():,} ({customer_last_purchase['churned'].mean()*100:.1f}%)")
print(f"  Active customers: {(~customer_last_purchase['churned'].astype(bool)).sum():,} ({(1-customer_last_purchase['churned'].mean())*100:.1f}%)")

# Merge with RFM data
churn_data = rfm[['customer_id', 'recency', 'frequency', 'monetary', 'segment']].merge(
    customer_last_purchase[['customer_id', 'churned']], 
    on='customer_id'
)

# Add customer features
customer_features = trans_full.groupby('customer_id').agg({
    'discount_applied': 'mean',
    'age': 'first',
    'gender': 'first',
    'loyalty_tier': 'first',
    'acquisition_channel': 'first'
}).reset_index()

churn_data = churn_data.merge(customer_features, on='customer_id')

# Encode categorical variables
le_gender = LabelEncoder()
le_loyalty = LabelEncoder()
le_channel = LabelEncoder()
le_segment = LabelEncoder()

churn_data['gender_encoded'] = le_gender.fit_transform(churn_data['gender'])
churn_data['loyalty_encoded'] = le_loyalty.fit_transform(churn_data['loyalty_tier'])
churn_data['channel_encoded'] = le_channel.fit_transform(churn_data['acquisition_channel'])
churn_data['segment_encoded'] = le_segment.fit_transform(churn_data['segment'])

# Prepare features for modeling
feature_cols = [
    'recency', 'frequency', 'monetary', 'discount_applied', 'age',
    'gender_encoded', 'loyalty_encoded', 'channel_encoded', 'segment_encoded'
]

X = churn_data[feature_cols].fillna(0)
y = churn_data['churned']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nModel Training:")
print(f"  Training set: {len(X_train):,} customers")
print(f"  Test set: {len(X_test):,} customers")

# Train Random Forest model with regularization to avoid overfitting
rf_churn = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42
)

rf_churn.fit(X_train, y_train)

# Make predictions
y_pred = rf_churn.predict(X_test)
y_prob = rf_churn.predict_proba(X_test)[:, 1]

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob)
except:
    auc = 0

print(f"\nModel Performance:")
print(f"  Accuracy: {accuracy*100:.1f}%")
if auc > 0:
    print(f"  AUC-ROC: {auc:.3f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Active', 'At-Risk/Churned']))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_churn.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Churn Predictors:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# Predict churn probability for all customers
churn_data['churn_probability'] = rf_churn.predict_proba(X.fillna(0))[:, 1]

# Identify high-risk customers
high_risk = churn_data[churn_data['churn_probability'] > 0.6]
high_risk_value = high_risk['monetary'].sum()

print(f"\nHigh-Risk Customer Analysis:")
print(f"  High-risk customers (>60% churn probability): {len(high_risk):,}")
print(f"  At-risk revenue value: ${high_risk_value:,.2f}")
print(f"  Average value per at-risk customer: ${high_risk['monetary'].mean():.2f}")

# Save churn predictions
churn_data[['customer_id', 'segment', 'churned', 'churn_probability', 'monetary']].to_csv(
    'analysis_churn_predictions_fixed.csv', index=False
)
print(f"\n✓ Saved: analysis_churn_predictions_fixed.csv")

print("\n" + "=" * 70)
print("\nOptimizing budget allocation...")

# Budget Optimization
# Estimate budget based on revenue (assuming 25% marketing spend)
campaigns_perf['estimated_budget'] = campaigns_perf['total_revenue'] * 0.25

# Calculate channel-level performance
channel_performance = campaigns_perf.groupby('channel').agg({
    'estimated_budget': 'sum',
    'total_revenue': 'sum',
    'unique_customers': 'sum',
    'total_transactions': 'sum'
}).reset_index()

# Calculate key metrics
channel_performance['roas'] = (
    channel_performance['total_revenue'] / channel_performance['estimated_budget']
)
channel_performance['cac'] = (
    channel_performance['estimated_budget'] / channel_performance['unique_customers']
)
channel_performance['cpt'] = (
    channel_performance['estimated_budget'] / channel_performance['total_transactions']
)

print(f"\nCurrent Channel Performance:")
print(channel_performance[['channel', 'estimated_budget', 'total_revenue', 'roas', 'cac']].round(2))

# Optimization: Allocate budget based on ROAS^2 (exponential preference for high performers)
total_budget = channel_performance['estimated_budget'].sum()

channel_performance['roas_weight'] = channel_performance['roas'] ** 2
total_weight = channel_performance['roas_weight'].sum()

channel_performance['optimized_budget'] = (
    channel_performance['roas_weight'] / total_weight * total_budget
)

channel_performance['budget_change'] = (
    channel_performance['optimized_budget'] - channel_performance['estimated_budget']
)

channel_performance['budget_change_pct'] = (
    channel_performance['budget_change'] / channel_performance['estimated_budget'] * 100
)

# Project revenue with optimized budget
channel_performance['projected_revenue'] = (
    channel_performance['optimized_budget'] * channel_performance['roas']
)

channel_performance['revenue_increase'] = (
    channel_performance['projected_revenue'] - channel_performance['total_revenue']
)

print(f"\nOptimized Budget Allocation:")
print(channel_performance[['channel', 'optimized_budget', 'budget_change_pct', 
                          'projected_revenue', 'revenue_increase']].round(2))

# Calculate total impact
total_revenue_increase = channel_performance['revenue_increase'].sum()
current_total_revenue = channel_performance['total_revenue'].sum()
roi_improvement = (total_revenue_increase / current_total_revenue) * 100

print(f"\nOptimization Impact:")
print(f"  Current total revenue: ${current_total_revenue:,.2f}")
print(f"  Projected total revenue: ${current_total_revenue + total_revenue_increase:,.2f}")
print(f"  Revenue increase: ${total_revenue_increase:,.2f}")
print(f"  ROI improvement: {roi_improvement:.1f}%")

# Save budget optimization
channel_performance.to_csv('analysis_budget_optimization_fixed.csv', index=False)
print(f"\n✓ Saved: analysis_budget_optimization_fixed.csv")

print("\n" + "=" * 70)
print("\nKey Insights Summary:")
print("=" * 70)
print(f"• Multi-touch customers: {multi_touch_pct:.1f}% of base")
print(f"• Revenue uplift (multi-touch): {uplift:+.1f}%")
print(f"• Churn model accuracy: {accuracy*100:.1f}%")
print(f"• High-risk customers: {len(high_risk):,} worth ${high_risk_value:,.0f}")
print(f"• Budget optimization: {roi_improvement:.1f}% projected ROI improvement")
print(f"• Revenue opportunity: ${total_revenue_increase:,.0f}")
print(f"• Top churn predictor: {feature_importance.iloc[0]['feature']}")
print("=" * 70)

print("\n✅ Advanced analysis complete!")
print("Next: Run events_analysis.py for funnel and behavior analysis\n")