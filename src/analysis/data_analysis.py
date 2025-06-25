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

def save_plot(fig, filename):
    """Save the current figure to a PDF file"""
    plt.savefig(PLOTS_DIR / filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {PLOTS_DIR / filename}")

def analyze_ride_types(df):
    """Analyze ride type preferences and patterns"""
    print("\n=== Ride Type Analysis ===")
    
    # Analyze ride type preferences
    ride_type_columns = [col for col in df.columns if 'proportion_of_' in col and '_rides_' in col]
    
    # Calculate average proportions
    ride_type_means = df[ride_type_columns].mean().sort_values(ascending=False)
    
    fig = plt.figure(figsize=(12, 6))
    ride_type_means.plot(kind='bar')
    plt.title('Average Proportion of Different Ride Types')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, 'ride_type_proportions.pdf')
    
    return ride_type_means

def analyze_prices(df):
    """Analyze price differences between ride types"""
    print("\n=== Price Analysis ===")
    
    # Analyze price differences between ride types
    price_diff_columns = [col for col in df.columns if 'price_diff_wrt_standard' in col]
    
    fig = plt.figure(figsize=(12, 6))
    df[price_diff_columns].boxplot()
    plt.title('Price Differences Relative to Standard Rides')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, 'price_differences.pdf')
    
    return df[price_diff_columns].describe()

def analyze_etas(df):
    """Analyze ETA differences between ride types"""
    print("\n=== ETA Analysis ===")
    
    # Analyze ETA differences between ride types
    eta_diff_columns = [col for col in df.columns if 'pin_eta_diff_wrt_standard' in col]
    
    fig = plt.figure(figsize=(12, 6))
    df[eta_diff_columns].boxplot()
    plt.title('ETA Differences Relative to Standard Rides (minutes)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, 'eta_differences.pdf')
    
    return df[eta_diff_columns].describe()

def analyze_user_behavior(df):
    """Analyze user behavior metrics"""
    print("\n=== User Behavior Analysis ===")
    
    # Analyze user behavior metrics
    behavior_metrics = [
        'total_rides_28d',
        'total_rides_90d',
        'unique_ride_active_days_28d',
        'unique_ride_active_days_91d',
        'ride_cancel_rate_28d',
        'ride_cancel_rate_91d'
    ]
    
    fig = plt.figure(figsize=(15, 10))
    for i, metric in enumerate(behavior_metrics, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data=df, x=metric, bins=50)
        plt.title(f'Distribution of {metric}')
    plt.tight_layout()
    save_plot(fig, 'user_behavior_metrics.pdf')
    
    return df[behavior_metrics].describe()

def analyze_correlations(df):
    """Analyze correlations between numeric features"""
    print("\n=== Correlation Analysis ===")
    
    # Select numeric columns for correlation analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_columns].corr()
    
    # Plot correlation heatmap
    fig = plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    save_plot(fig, 'correlation_matrix.pdf')
    
    return correlation_matrix

def analyze_time_series(df):
    """Analyze time-based patterns"""
    print("\n=== Time-based Analysis ===")
    
    # Convert ds to datetime if it's not already
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Group by date and calculate daily metrics
    daily_metrics = df.groupby('ds').agg({
        'total_rides_28d': 'mean',
        'standard_final_price_major_currency': 'mean',
        'estimated_travel_time_sec': 'mean'
    }).reset_index()
    
    # Plot daily trends
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    ax1.plot(daily_metrics['ds'], daily_metrics['total_rides_28d'])
    ax1.set_title('Daily Average Total Rides')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.plot(daily_metrics['ds'], daily_metrics['standard_final_price_major_currency'])
    ax2.set_title('Daily Average Standard Price')
    ax2.tick_params(axis='x', rotation=45)
    
    ax3.plot(daily_metrics['ds'], daily_metrics['estimated_travel_time_sec'] / 60)  # Convert to minutes
    ax3.set_title('Daily Average Travel Time (minutes)')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_plot(fig, 'time_series_analysis.pdf')
    
    return daily_metrics

def main():
    # Load the data
    print("Loading data...")
    df = load_parquet_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    df.info()
    
    print("\nSample of the data:")
    print(df.head())
    
    # Run all analyses
    ride_type_stats = analyze_ride_types(df)
    price_stats = analyze_prices(df)
    eta_stats = analyze_etas(df)
    behavior_stats = analyze_user_behavior(df)
    correlation_matrix = analyze_correlations(df)
    time_series_stats = analyze_time_series(df)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nRide Type Proportions:")
    print(ride_type_stats)
    
    print("\nPrice Differences:")
    print(price_stats)
    
    print("\nETA Differences:")
    print(eta_stats)
    
    print("\nUser Behavior Metrics:")
    print(behavior_stats)
    
    print("\nTime Series Statistics:")
    print(time_series_stats.describe())

if __name__ == "__main__":
    main() 