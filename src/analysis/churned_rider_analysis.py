import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os
from load_data import load_parquet_data
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn')
sns.set_palette('husl')

# Create plots directory
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

def identify_churned_riders(df):
    """Identify churned riders based on criteria:
    - Signed up >365 days ago
    - >2 lifetime rides
    - Zero rides in last 365 days
    """
    print("\n=== Identifying Churned Riders ===")
    
    # Convert ds to datetime if not already
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Calculate days since signup
    #df['days_since_signup'] = (df['ds'] - pd.to_datetime(df['signup_date'])).dt.days
    
    # Identify churned riders
    churned_mask = (
        (df['days_since_signup'] > 365) &  # Signed up >365 days ago
        (df['rides_lifetime'] > 2) &      # >2 lifetime rides
        (df['all_type_total_rides_365d'] == 0)       # Zero rides in last 28 days (as proxy for 365)
    )
    
    churned_riders = df[churned_mask]
    print(f"Found {len(churned_riders)} churned riders")
    
    return churned_riders

def analyze_mode_preferences(df, churned_riders):
    """Analyze mode preferences for churned riders based on requested_ride_type"""
    print("\n=== Analyzing Mode Preferences ===")
    
    # Count most common requested_ride_type for churned riders
    churned_type_counts = churned_riders['requested_ride_type'].value_counts().sort_values(ascending=False)
    
    # Count most common requested_ride_type for all riders
    overall_type_counts = df['requested_ride_type'].value_counts().sort_values(ascending=False)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Subplot 1: Top 5 for churned riders
    churned_top5 = churned_type_counts.head(5)
    bars1 = axes[0].bar(churned_top5.index, churned_top5.values, color='pink')
    axes[0].set_title('Top 5 Requested Ride Types for Churned Riders')
    axes[0].set_ylabel('Number of Rides')
    axes[0].set_xticklabels(churned_top5.index, rotation=45, ha='right')
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 2: Top 5 for all riders
    overall_top5 = overall_type_counts.head(5)
    bars2 = axes[1].bar(overall_top5.index, overall_top5.values, color='goldenrod')
    axes[1].set_title('Top 5 Requested Ride Types for All Riders')
    axes[1].set_ylabel('Number of Rides')
    axes[1].set_xticklabels(overall_top5.index, rotation=45, ha='right')
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[1].annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to PDF
    plt.savefig(PLOTS_DIR / 'churned_rider_mode_preferences.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved mode preference analysis to {PLOTS_DIR / 'churned_rider_mode_preferences.pdf'}")
    
    return churned_type_counts, overall_type_counts

def main():
    # Load the data
    print("Loading data...")
    df = load_parquet_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Identify churned riders
    churned_riders = identify_churned_riders(df)
    
    # Analyze mode preferences
    churned_type_counts, overall_type_counts = analyze_mode_preferences(df, churned_riders)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nTop 5 Requested Ride Types for Churned Riders:")
    print(churned_type_counts.head())

if __name__ == "__main__":
    main() 