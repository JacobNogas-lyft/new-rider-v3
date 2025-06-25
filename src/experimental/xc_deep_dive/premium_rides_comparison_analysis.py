import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Add the src directory to the path so we can import load_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.load_data import load_parquet_data

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load the data and prepare for analysis."""
    print("Loading data...")
    df = load_parquet_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def analyze_premium_rides_columns(df):
    """Analyze the premium rides columns."""
    print("\n" + "="*60)
    print("PREMIUM RIDES COLUMNS ANALYSIS")
    print("="*60)
    
    # Check if columns exist
    lifetime_col = 'rides_premium_lifetime'
    recent_col = 'extra_comfort_total_rides_365d'
    
    if lifetime_col not in df.columns:
        print(f"ERROR: '{lifetime_col}' column not found in dataset!")
        return None, None
    
    if recent_col not in df.columns:
        print(f"ERROR: '{recent_col}' column not found in dataset!")
        return None, None
    
    # Get the columns
    lifetime_rides = df[lifetime_col]
    recent_rides = df[recent_col]
    
    print(f"Column 1: {lifetime_col}")
    print(f"  Data type: {lifetime_rides.dtype}")
    print(f"  Total rows: {len(lifetime_rides):,}")
    print(f"  Non-null values: {lifetime_rides.count():,}")
    print(f"  Null values: {lifetime_rides.isnull().sum():,}")
    print(f"  Null percentage: {(lifetime_rides.isnull().sum() / len(lifetime_rides)) * 100:.2f}%")
    
    print(f"\nColumn 2: {recent_col}")
    print(f"  Data type: {recent_rides.dtype}")
    print(f"  Total rows: {len(recent_rides):,}")
    print(f"  Non-null values: {recent_rides.count():,}")
    print(f"  Null values: {recent_rides.isnull().sum():,}")
    print(f"  Null percentage: {(recent_rides.isnull().sum() / len(recent_rides)) * 100:.2f}%")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics - {lifetime_col}:")
    print(f"  Mean: {lifetime_rides.mean():.2f}")
    print(f"  Median: {lifetime_rides.median():.2f}")
    print(f"  Std: {lifetime_rides.std():.2f}")
    print(f"  Min: {lifetime_rides.min()}")
    print(f"  Max: {lifetime_rides.max()}")
    print(f"  25th percentile: {lifetime_rides.quantile(0.25):.2f}")
    print(f"  75th percentile: {lifetime_rides.quantile(0.75):.2f}")
    
    print(f"\nDescriptive Statistics - {recent_col}:")
    print(f"  Mean: {recent_rides.mean():.2f}")
    print(f"  Median: {recent_rides.median():.2f}")
    print(f"  Std: {recent_rides.std():.2f}")
    print(f"  Min: {recent_rides.min()}")
    print(f"  Max: {recent_rides.max()}")
    print(f"  25th percentile: {recent_rides.quantile(0.25):.2f}")
    print(f"  75th percentile: {recent_rides.quantile(0.75):.2f}")
    
    return lifetime_rides, recent_rides

def analyze_relationship(df, lifetime_rides, recent_rides):
    """Analyze the relationship between lifetime and recent premium rides."""
    print("\n" + "="*60)
    print("RELATIONSHIP ANALYSIS")
    print("="*60)
    
    # Correlation analysis
    correlation = lifetime_rides.corr(recent_rides)
    print(f"Pearson correlation: {correlation:.4f}")
    
    # Spearman correlation (for non-linear relationships)
    spearman_corr = lifetime_rides.corr(recent_rides, method='spearman')
    print(f"Spearman correlation: {spearman_corr:.4f}")
    
    # Create comparison metrics
    df['premium_ratio'] = recent_rides / (lifetime_rides + 1)  # Add 1 to avoid division by zero
    df['premium_difference'] = lifetime_rides - recent_rides
    
    print(f"\nRatio Analysis (recent/lifetime):")
    print(f"  Mean ratio: {df['premium_ratio'].mean():.4f}")
    print(f"  Median ratio: {df['premium_ratio'].median():.4f}")
    print(f"  Std ratio: {df['premium_ratio'].std():.4f}")
    
    print(f"\nDifference Analysis (lifetime - recent):")
    print(f"  Mean difference: {df['premium_difference'].mean():.2f}")
    print(f"  Median difference: {df['premium_difference'].median():.2f}")
    print(f"  Std difference: {df['premium_difference'].std():.2f}")
    
    # Find cases where recent is low but lifetime is high
    print(f"\nCases with low recent but high lifetime premium rides:")
    
    # Define thresholds
    low_recent_threshold = 2  # Low recent rides
    high_lifetime_threshold = 10  # High lifetime rides
    
    low_recent_high_lifetime = df[
        (recent_rides <= low_recent_threshold) & 
        (lifetime_rides >= high_lifetime_threshold)
    ]
    
    print(f"  Riders with recent <= {low_recent_threshold} AND lifetime >= {high_lifetime_threshold}:")
    print(f"    Count: {len(low_recent_high_lifetime):,}")
    print(f"    Percentage: {len(low_recent_high_lifetime) / len(df) * 100:.2f}%")
    
    if len(low_recent_high_lifetime) > 0:
        print(f"    Average lifetime rides: {low_recent_high_lifetime['rides_premium_lifetime'].mean():.2f}")
        print(f"    Average recent rides: {low_recent_high_lifetime['extra_comfort_total_rides_365d'].mean():.2f}")
        print(f"    Average ratio: {low_recent_high_lifetime['premium_ratio'].mean():.4f}")
    
    # More detailed breakdown
    print(f"\nDetailed breakdown by recent rides:")
    for recent_threshold in [0, 1, 2, 3, 4, 5]:
        subset = df[recent_rides <= recent_threshold]
        if len(subset) > 0:
            avg_lifetime = subset['rides_premium_lifetime'].mean()
            count = len(subset)
            print(f"  Recent <= {recent_threshold}: {count:,} riders, avg lifetime: {avg_lifetime:.2f}")
    
    return df

def identify_churned_premium_users(df, lifetime_rides, recent_rides):
    """Identify users who might be 'churned' from premium rides."""
    print("\n" + "="*60)
    print("CHURNED PREMIUM USERS ANALYSIS")
    print("="*60)
    
    # Define churned premium users: high lifetime, low recent
    churned_thresholds = [
        (5, 20),   # Recent <= 5, Lifetime >= 20
        (3, 15),   # Recent <= 3, Lifetime >= 15
        (2, 10),   # Recent <= 2, Lifetime >= 10
        (1, 5),    # Recent <= 1, Lifetime >= 5
        (0, 3),    # Recent = 0, Lifetime >= 3
    ]
    
    for recent_thresh, lifetime_thresh in churned_thresholds:
        churned_users = df[
            (recent_rides <= recent_thresh) & 
            (lifetime_rides >= lifetime_thresh)
        ]
        
        if len(churned_users) > 0:
            print(f"\nChurned Premium Users (Recent <= {recent_thresh}, Lifetime >= {lifetime_thresh}):")
            print(f"  Count: {len(churned_users):,}")
            print(f"  Percentage of total: {len(churned_users) / len(df) * 100:.2f}%")
            print(f"  Average lifetime rides: {churned_users['rides_premium_lifetime'].mean():.2f}")
            print(f"  Average recent rides: {churned_users['extra_comfort_total_rides_365d'].mean():.2f}")
            print(f"  Average ratio (recent/lifetime): {churned_users['premium_ratio'].mean():.4f}")
            
            # Check if they're still using other ride types
            if 'all_type_total_rides_365d' in df.columns:
                other_rides = churned_users['all_type_total_rides_365d'] - churned_users['extra_comfort_total_rides_365d']
                print(f"  Average other rides in 365d: {other_rides.mean():.2f}")
                still_active = (other_rides > 0).sum()
                print(f"  Still active with other ride types: {still_active:,} ({still_active/len(churned_users)*100:.1f}%)")

def plot_comparison_analysis(df, lifetime_rides, recent_rides, save_path=None):
    """Create comprehensive plots comparing lifetime vs recent premium rides."""
    print("\nCreating comparison analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Premium Rides: Lifetime vs Recent (365d) Comparison', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot
    axes[0, 0].scatter(recent_rides, lifetime_rides, alpha=0.3, s=1)
    axes[0, 0].set_xlabel('Extra Comfort Total Rides (365d)')
    axes[0, 0].set_ylabel('Premium Rides (Lifetime)')
    axes[0, 0].set_title('Scatter Plot: Recent vs Lifetime')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add diagonal line for reference
    max_val = max(lifetime_rides.max(), recent_rides.max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='y=x')
    axes[0, 0].legend()
    
    # 2. Log scale scatter plot
    # Filter out zeros for log scale
    non_zero_mask = (recent_rides > 0) & (lifetime_rides > 0)
    if non_zero_mask.sum() > 0:
        axes[0, 1].scatter(recent_rides[non_zero_mask], lifetime_rides[non_zero_mask], alpha=0.3, s=1)
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_xlabel('Extra Comfort Total Rides (365d) - Log Scale')
        axes[0, 1].set_ylabel('Premium Rides (Lifetime) - Log Scale')
        axes[0, 1].set_title('Log Scale Scatter Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add diagonal line
        min_val = min(recent_rides[non_zero_mask].min(), lifetime_rides[non_zero_mask].min())
        max_val = max(recent_rides[non_zero_mask].max(), lifetime_rides[non_zero_mask].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        axes[0, 1].legend()
    
    # 3. Ratio distribution
    ratio_data = df['premium_ratio']
    axes[0, 2].hist(ratio_data, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 2].set_xlabel('Ratio (Recent/Lifetime)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Recent/Lifetime Ratio')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Difference distribution
    diff_data = df['premium_difference']
    axes[1, 0].hist(diff_data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Difference (Lifetime - Recent)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Lifetime - Recent Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    axes[1, 1].boxplot([lifetime_rides, recent_rides], labels=['Lifetime', 'Recent (365d)'])
    axes[1, 1].set_ylabel('Number of Premium Rides')
    axes[1, 1].set_title('Box Plot Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Churned users analysis
    # Define churned users (recent <= 2, lifetime >= 10)
    churned_mask = (recent_rides <= 2) & (lifetime_rides >= 10)
    active_mask = ~churned_mask
    
    churned_lifetime = lifetime_rides[churned_mask]
    churned_recent = recent_rides[churned_mask]
    active_lifetime = lifetime_rides[active_mask]
    active_recent = recent_rides[active_mask]
    
    axes[1, 2].scatter(active_recent, active_lifetime, alpha=0.3, s=1, color='blue', label='Active Users')
    axes[1, 2].scatter(churned_recent, churned_lifetime, alpha=0.7, s=5, color='red', label='Churned Users')
    axes[1, 2].set_xlabel('Extra Comfort Total Rides (365d)')
    axes[1, 2].set_ylabel('Premium Rides (Lifetime)')
    axes[1, 2].set_title('Active vs Churned Premium Users')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison analysis plots saved to: {save_path}")
    
    plt.show()

def plot_churned_users_analysis(df, lifetime_rides, recent_rides, save_path=None):
    """Create detailed analysis of churned premium users."""
    print("\nCreating churned users analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Churned Premium Users Analysis', fontsize=16, fontweight='bold')
    
    # Define churned users (recent <= 2, lifetime >= 10)
    churned_mask = (recent_rides <= 2) & (lifetime_rides >= 10)
    churned_users = df[churned_mask]
    
    if len(churned_users) == 0:
        print("No churned users found with the specified criteria.")
        return
    
    # 1. Distribution of lifetime rides for churned users
    axes[0, 0].hist(churned_users['rides_premium_lifetime'], bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Lifetime Premium Rides')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Lifetime Premium Rides Distribution\n(Churned Users)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution of recent rides for churned users
    axes[0, 1].hist(churned_users['extra_comfort_total_rides_365d'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_xlabel('Recent Extra Comfort Rides (365d)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Recent Extra Comfort Rides Distribution\n(Churned Users)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Ratio distribution for churned users
    axes[1, 0].hist(churned_users['premium_ratio'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].set_xlabel('Ratio (Recent/Lifetime)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Recent/Lifetime Ratio Distribution\n(Churned Users)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter plot highlighting churned users
    all_users = df[~churned_mask]  # Non-churned users
    
    axes[1, 1].scatter(all_users['extra_comfort_total_rides_365d'], all_users['rides_premium_lifetime'], 
                       alpha=0.1, s=1, color='lightblue', label='Active Users')
    axes[1, 1].scatter(churned_users['extra_comfort_total_rides_365d'], churned_users['rides_premium_lifetime'], 
                       alpha=0.8, s=10, color='red', label='Churned Users')
    axes[1, 1].set_xlabel('Extra Comfort Total Rides (365d)')
    axes[1, 1].set_ylabel('Premium Rides (Lifetime)')
    axes[1, 1].set_title('Churned vs Active Premium Users')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        churned_save_path = str(save_path).replace('.png', '_churned_users.png')
        plt.savefig(churned_save_path, dpi=300, bbox_inches='tight')
        print(f"Churned users analysis plots saved to: {churned_save_path}")
    
    plt.show()

def plot_threshold_analysis(df, lifetime_rides, recent_rides, save_path=None):
    """Analyze different thresholds for identifying churned users."""
    print("\nCreating threshold analysis plots...")
    
    # Define different thresholds
    recent_thresholds = [0, 1, 2, 3, 4, 5]
    lifetime_thresholds = [5, 10, 15, 20, 25]
    
    # Create heatmap data
    heatmap_data = []
    
    print("Churned users count by different thresholds:")
    print("Recent\\Lifetime", end="\t")
    for lt in lifetime_thresholds:
        print(f"≥{lt}", end="\t")
    print()
    
    for rt in recent_thresholds:
        row = []
        print(f"≤{rt}", end="\t")
        for lt in lifetime_thresholds:
            count = len(df[(recent_rides <= rt) & (lifetime_rides >= lt)])
            row.append(count)
            print(f"{count:,}", end="\t")
        heatmap_data.append(row)
        print()
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Heatmap
    im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(lifetime_thresholds)))
    ax1.set_yticks(range(len(recent_thresholds)))
    ax1.set_xticklabels([f'≥{lt}' for lt in lifetime_thresholds])
    ax1.set_yticklabels([f'≤{rt}' for rt in recent_thresholds])
    ax1.set_xlabel('Lifetime Premium Rides Threshold')
    ax1.set_ylabel('Recent Premium Rides Threshold')
    ax1.set_title('Churned Users Count by Thresholds')
    
    # Add text annotations
    for i in range(len(recent_thresholds)):
        for j in range(len(lifetime_thresholds)):
            text = ax1.text(j, i, f'{heatmap_data[i][j]:,}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax1)
    
    # Plot 2: Percentage heatmap
    total_users = len(df)
    percentage_data = [[count/total_users*100 for count in row] for row in heatmap_data]
    
    im2 = ax2.imshow(percentage_data, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(lifetime_thresholds)))
    ax2.set_yticks(range(len(recent_thresholds)))
    ax2.set_xticklabels([f'≥{lt}' for lt in lifetime_thresholds])
    ax2.set_yticklabels([f'≤{rt}' for rt in recent_thresholds])
    ax2.set_xlabel('Lifetime Premium Rides Threshold')
    ax2.set_ylabel('Recent Premium Rides Threshold')
    ax2.set_title('Churned Users Percentage by Thresholds')
    
    # Add text annotations
    for i in range(len(recent_thresholds)):
        for j in range(len(lifetime_thresholds)):
            text = ax2.text(j, i, f'{percentage_data[i][j]:.1f}%',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    if save_path:
        threshold_save_path = str(save_path).replace('.png', '_threshold_analysis.png')
        plt.savefig(threshold_save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold analysis plots saved to: {threshold_save_path}")
    
    plt.show()

def main():
    """Main function to run the analysis."""
    try:
        # Load data
        df = load_and_prepare_data()
        
        # Analyze premium rides columns
        lifetime_rides, recent_rides = analyze_premium_rides_columns(df)
        
        if lifetime_rides is not None and recent_rides is not None:
            # Analyze relationship
            df = analyze_relationship(df, lifetime_rides, recent_rides)
            
            # Identify churned premium users
            identify_churned_premium_users(df, lifetime_rides, recent_rides)
            
            # Create plots directory
            plots_dir = Path('/home/sagemaker-user/studio/src/new-rider-v3/plots/xc_deep_dive')
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            # Create plots
            comparison_path = plots_dir / 'premium_rides_comparison.png'
            plot_comparison_analysis(df, lifetime_rides, recent_rides, save_path=comparison_path)
            plot_churned_users_analysis(df, lifetime_rides, recent_rides, save_path=comparison_path)
            plot_threshold_analysis(df, lifetime_rides, recent_rides, save_path=comparison_path)
            
            print(f"\nAnalysis completed!")
            print(f"Plots saved to: {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 