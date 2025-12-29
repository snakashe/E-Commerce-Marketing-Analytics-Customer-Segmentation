"""
Customer Journey & Funnel Analysis
Analyzes customer behavior using events data
Author: Sakshi Nakashe
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("\nCustomer Journey & Funnel Analysis")
print("=" * 70)
print("\nLoading events data...")

# Load events data
try:
    events = pd.read_csv('events.csv')
    print(f"✓ Events loaded: {len(events):,} records")
except FileNotFoundError:
    print("\n❌ Error: events.csv not found")
    print("Please ensure events.csv is in the same directory as this script.")
    exit()

# Convert timestamp
events['timestamp'] = pd.to_datetime(events['timestamp'])

# Data overview
print(f"\nDataset Overview:")
print(f"  Date range: {events['timestamp'].min().date()} to {events['timestamp'].max().date()}")
print(f"  Unique customers: {events['customer_id'].nunique():,}")
print(f"  Unique sessions: {events['session_id'].nunique():,}")

print("\n" + "=" * 70)
print("\nAnalyzing conversion funnel...")

# Conversion Funnel Analysis
total_sessions = events['session_id'].nunique()
total_customers = events['customer_id'].nunique()

# Count events by type
event_counts = events['event_type'].value_counts()

print(f"\nFunnel Overview:")
print(f"  Total sessions: {total_sessions:,}")
print(f"  Total customers: {total_customers:,}")
print(f"  Total events: {len(events):,}")

# Extract funnel stage counts
view_count = event_counts.get('view', 0)
click_count = event_counts.get('click', 0)
add_to_cart_count = event_counts.get('add_to_cart', 0)
purchase_count = event_counts.get('purchase', 0)
bounce_count = event_counts.get('bounce', 0)

print(f"\nFunnel Stages:")
if view_count > 0:
    print(f"  View: {view_count:,} ({view_count/len(events)*100:.1f}% of events)")
    
    if click_count > 0:
        click_rate = (click_count / view_count) * 100
        print(f"  Click: {click_count:,} ({click_rate:.1f}% of views)")
    
    if add_to_cart_count > 0:
        if click_count > 0:
            cart_rate = (add_to_cart_count / click_count) * 100
            print(f"  Add to Cart: {add_to_cart_count:,} ({cart_rate:.1f}% of clicks)")
    
    if purchase_count > 0:
        if add_to_cart_count > 0:
            purchase_rate = (purchase_count / add_to_cart_count) * 100
            print(f"  Purchase: {purchase_count:,} ({purchase_rate:.1f}% of carts)")
        
        overall_conversion = (purchase_count / view_count) * 100
        print(f"\n  Overall Conversion Rate: {overall_conversion:.2f}%")

if bounce_count > 0:
    bounce_rate = (bounce_count / len(events)) * 100
    print(f"  Bounce: {bounce_count:,} ({bounce_rate:.1f}% of events)")

print("\n" + "=" * 70)
print("\nAnalyzing channel performance...")

# Channel Performance Analysis
channel_events = events[events['traffic_source'].notna()].groupby('traffic_source').agg({
    'event_id': 'count',
    'session_id': 'nunique',
    'customer_id': 'nunique'
}).reset_index()

channel_events.columns = ['traffic_source', 'total_events', 'sessions', 'customers']

# Calculate conversions by channel
channel_conversions = events[events['event_type'] == 'purchase'].groupby('traffic_source').size().reset_index(name='conversions')
channel_events = channel_events.merge(channel_conversions, on='traffic_source', how='left')
channel_events['conversions'] = channel_events['conversions'].fillna(0).astype(int)

# Calculate conversion rate
channel_events['conversion_rate'] = (channel_events['conversions'] / channel_events['sessions'] * 100).round(2)
channel_events = channel_events.sort_values('conversion_rate', ascending=False)

print(f"\nChannel Conversion Rates:")
for _, row in channel_events.iterrows():
    print(f"  {row['traffic_source']}: {row['conversion_rate']:.2f}% ({int(row['conversions'])} purchases from {int(row['sessions'])} sessions)")

print("\n" + "=" * 70)
print("\nAnalyzing device performance...")

# Device Performance Analysis
device_events = events[events['device_type'].notna()].groupby('device_type').agg({
    'event_id': 'count',
    'session_id': 'nunique'
}).reset_index()

device_events.columns = ['device_type', 'total_events', 'sessions']

# Device conversions
device_conversions = events[events['event_type'] == 'purchase'].groupby('device_type').size().reset_index(name='conversions')
device_events = device_events.merge(device_conversions, on='device_type', how='left')
device_events['conversions'] = device_events['conversions'].fillna(0).astype(int)

# Calculate conversion rate
device_events['conversion_rate'] = (device_events['conversions'] / device_events['sessions'] * 100).round(2)
device_events = device_events.sort_values('conversion_rate', ascending=False)

print(f"\nDevice Conversion Rates:")
for _, row in device_events.iterrows():
    print(f"  {row['device_type']}: {row['conversion_rate']:.2f}% ({int(row['conversions'])} purchases)")

print("\n" + "=" * 70)
print("\nAnalyzing A/B test results...")

# A/B Test Analysis
if 'experiment_group' in events.columns:
    ab_test = events.groupby('experiment_group').agg({
        'event_id': 'count',
        'session_id': 'nunique'
    }).reset_index()
    
    ab_test.columns = ['experiment_group', 'total_events', 'sessions']
    
    # A/B test conversions
    ab_conversions = events[events['event_type'] == 'purchase'].groupby('experiment_group').size().reset_index(name='conversions')
    ab_test = ab_test.merge(ab_conversions, on='experiment_group', how='left')
    ab_test['conversions'] = ab_test['conversions'].fillna(0).astype(int)
    
    # Calculate conversion rate
    ab_test['conversion_rate'] = (ab_test['conversions'] / ab_test['sessions'] * 100).round(2)
    ab_test = ab_test.sort_values('conversion_rate', ascending=False)
    
    print(f"\nExperiment Group Results:")
    for _, row in ab_test.iterrows():
        print(f"  {row['experiment_group']}: {row['conversion_rate']:.2f}% conversion ({int(row['conversions'])} purchases)")
    
    # Calculate uplift vs control
    control = ab_test[ab_test['experiment_group'] == 'Control']
    if len(control) > 0:
        control_rate = control['conversion_rate'].values[0]
        
        print(f"\n  Performance vs Control:")
        for _, row in ab_test.iterrows():
            if row['experiment_group'] != 'Control':
                diff = row['conversion_rate'] - control_rate
                print(f"    {row['experiment_group']}: {diff:+.2f} percentage points")
else:
    print("  No experiment group data found")

print("\n" + "=" * 70)
print("\nAnalyzing session engagement...")

# Session Engagement Analysis
session_metrics = events.groupby('session_id').agg({
    'event_id': 'count',
    'session_duration_sec': 'first',
    'event_type': lambda x: 1 if 'purchase' in x.values else 0
}).reset_index()

session_metrics.columns = ['session_id', 'events_per_session', 'duration', 'converted']

# Calculate averages
avg_events = session_metrics['events_per_session'].mean()
avg_duration = session_metrics['duration'].mean()
session_conversion_rate = (session_metrics['converted'].sum() / len(session_metrics) * 100)

print(f"\nSession Engagement Metrics:")
print(f"  Average events per session: {avg_events:.1f}")
print(f"  Average session duration: {avg_duration:.1f} seconds ({avg_duration/60:.1f} minutes)")
print(f"  Session conversion rate: {session_conversion_rate:.2f}%")

# Engagement by conversion status
converted_sessions = session_metrics[session_metrics['converted'] == 1]
non_converted_sessions = session_metrics[session_metrics['converted'] == 0]

if len(converted_sessions) > 0:
    print(f"\n  Converted sessions:")
    print(f"    Average events: {converted_sessions['events_per_session'].mean():.1f}")
    print(f"    Average duration: {converted_sessions['duration'].mean():.1f} seconds")

if len(non_converted_sessions) > 0:
    print(f"\n  Non-converted sessions:")
    print(f"    Average events: {non_converted_sessions['events_per_session'].mean():.1f}")
    print(f"    Average duration: {non_converted_sessions['duration'].mean():.1f} seconds")

print("\n" + "=" * 70)
print("\nAnalyzing bounce rates...")

# Bounce Rate Analysis
bounce_analysis = events[events['event_type'] == 'bounce'].groupby('traffic_source').size().reset_index(name='bounces')
total_sessions_by_source = events.groupby('traffic_source')['session_id'].nunique().reset_index(name='total_sessions')

bounce_analysis = bounce_analysis.merge(total_sessions_by_source, on='traffic_source')
bounce_analysis['bounce_rate'] = (bounce_analysis['bounces'] / bounce_analysis['total_sessions'] * 100).round(2)
bounce_analysis = bounce_analysis.sort_values('bounce_rate')

print(f"\nBounce Rates by Channel:")
for _, row in bounce_analysis.head(10).iterrows():
    print(f"  {row['traffic_source']}: {row['bounce_rate']:.2f}% ({int(row['bounces'])} bounces)")

print("\n" + "=" * 70)
print("\nSaving analysis results...")

# Save all analysis outputs
channel_events.to_csv('analysis_channel_funnel.csv', index=False)
device_events.to_csv('analysis_device_performance.csv', index=False)
session_metrics.to_csv('analysis_session_engagement.csv', index=False)
bounce_analysis.to_csv('analysis_bounce_rates.csv', index=False)

print(f"\n✓ analysis_channel_funnel.csv")
print(f"✓ analysis_device_performance.csv")
print(f"✓ analysis_session_engagement.csv")
print(f"✓ analysis_bounce_rates.csv")

if 'experiment_group' in events.columns:
    ab_test.to_csv('analysis_ab_test_results.csv', index=False)
    print(f"✓ analysis_ab_test_results.csv")

print("\n" + "=" * 70)
print("\nKey Insights Summary:")
print("=" * 70)

if view_count > 0 and purchase_count > 0:
    print(f"• Overall conversion rate: {(purchase_count/view_count)*100:.2f}%")

print(f"• Average session duration: {avg_duration:.1f} seconds")
print(f"• Average events per session: {avg_events:.1f}")

if len(channel_events) > 0:
    best_channel = channel_events.iloc[0]
    print(f"• Best converting channel: {best_channel['traffic_source']} ({best_channel['conversion_rate']:.2f}%)")

if len(device_events) > 0:
    best_device = device_events.iloc[0]
    print(f"• Best converting device: {best_device['device_type']} ({best_device['conversion_rate']:.2f}%)")

if len(converted_sessions) > 0 and len(non_converted_sessions) > 0:
    engagement_diff = converted_sessions['events_per_session'].mean() - non_converted_sessions['events_per_session'].mean()
    print(f"• Converted sessions have {engagement_diff:.1f} more events on average")

print("=" * 70)

print("\n✅ Events analysis complete!")
print("All analyses finished. Ready for Tableau visualization!\n")