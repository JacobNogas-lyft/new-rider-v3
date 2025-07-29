import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_parquet_data
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_eda_directory():
    """Create EDA directory if it doesn't exist."""
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
    eda_dir = base_path / 'reports' / 'EDA'
    eda_dir.mkdir(parents=True, exist_ok=True)
    return eda_dir

def load_and_prepare_data(data_version='v3'):
    """Load and prepare data for rides distribution analysis."""
    print(f"Loading data version: {data_version}")
    df = load_parquet_data(data_version)
    print(f"Data loaded: {len(df)} rows")
    return df

def analyze_feature_distribution(df, feature_name, eda_dir):
    """
    Analyze the distribution of a feature and save results.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        feature_name (str): Name of the feature to analyze
        eda_dir (Path): Directory to save results
    """
    print(f"\n=== Analyzing {feature_name} Distribution ===")
    
    # Check if feature exists
    if feature_name not in df.columns:
        print(f"Warning: '{feature_name}' column not found in data.")
        return None
    
    # Get feature data (drop NaN values)
    feature_data = df[feature_name].dropna()
    
    if len(feature_data) == 0:
        print(f"Warning: No valid data found for '{feature_name}'.")
        return None
    
    # Calculate descriptive statistics
    stats = {
        'feature_name': feature_name,
        'count': len(feature_data),
        'non_null_pct': (len(feature_data) / len(df)) * 100,
        'mean': feature_data.mean(),
        'std': feature_data.std(),
        'min': feature_data.min(),
        'max': feature_data.max(),
        'median': feature_data.median(),
        'q25': feature_data.quantile(0.25),
        'q75': feature_data.quantile(0.75),
        'q90': feature_data.quantile(0.90),
        'q95': feature_data.quantile(0.95),
        'q99': feature_data.quantile(0.99),
        'skewness': feature_data.skew(),
        'kurtosis': feature_data.kurtosis()
    }
    
    # Print statistics
    print(f"Count (non-null): {stats['count']:,}")
    print(f"Non-null percentage: {stats['non_null_pct']:.1f}%")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Standard deviation: {stats['std']:.2f}")
    print(f"Min: {stats['min']:.2f}")
    print(f"Max: {stats['max']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"25th percentile: {stats['q25']:.2f}")
    print(f"75th percentile: {stats['q75']:.2f}")
    print(f"90th percentile: {stats['q90']:.2f}")
    print(f"95th percentile: {stats['q95']:.2f}")
    print(f"99th percentile: {stats['q99']:.2f}")
    print(f"Skewness: {stats['skewness']:.2f}")
    print(f"Kurtosis: {stats['kurtosis']:.2f}")
    
    # Value counts for common values
    value_counts = feature_data.value_counts().head(20)
    print(f"\nTop 20 most common values:")
    for value, count in value_counts.items():
        pct = (count / len(feature_data)) * 100
        print(f"  {value}: {count:,} ({pct:.1f}%)")
    
    # Save descriptive statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_file = eda_dir / f'{feature_name}_descriptive_stats.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"\nSaved descriptive statistics to: {stats_file}")
    
    # Save value counts to CSV
    value_counts_df = pd.DataFrame({
        'value': value_counts.index,
        'count': value_counts.values,
        'percentage': (value_counts.values / len(feature_data)) * 100
    })
    value_counts_file = eda_dir / f'{feature_name}_value_counts.csv'
    value_counts_df.to_csv(value_counts_file, index=False)
    print(f"Saved value counts to: {value_counts_file}")
    
    return stats, feature_data

def create_distribution_plots(df, feature_name, feature_data, eda_dir):
    """
    Create distribution plots for a feature.
    
    Args:
        df (pandas.DataFrame): Full dataframe
        feature_name (str): Name of the feature
        feature_data (pandas.Series): Feature data (non-null values)
        eda_dir (Path): Directory to save plots
    """
    print(f"\nCreating distribution plots for {feature_name}...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Distribution Analysis: {feature_name}', fontsize=16, fontweight='bold')
    
    # 1. Histogram
    ax1 = axes[0, 0]
    ax1.hist(feature_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Histogram')
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    ax2.boxplot(feature_data, vert=True)
    ax2.set_title('Box Plot')
    ax2.set_ylabel(feature_name)
    ax2.grid(True, alpha=0.3)
    
    # 3. Log-scale histogram (if data has wide range)
    ax3 = axes[1, 0]
    if feature_data.max() > feature_data.min() * 10:  # Wide range
        # Filter out zeros for log scale
        log_data = feature_data[feature_data > 0]
        if len(log_data) > 0:
            ax3.hist(log_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_yscale('log')
            ax3.set_title('Histogram (Log Scale)')
            ax3.set_xlabel(feature_name)
            ax3.set_ylabel('Frequency (log scale)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No positive values for log scale', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Log Scale (No Data)')
    else:
        ax3.text(0.5, 0.5, 'Log scale not needed\n(narrow range)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Log Scale (Not Applicable)')
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_data = np.sort(feature_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax4.plot(sorted_data, cumulative, color='orange', linewidth=2)
    ax4.set_title('Cumulative Distribution')
    ax4.set_xlabel(feature_name)
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = eda_dir / f'{feature_name}_distribution_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plots to: {plot_file}")
    plt.close()

def analyze_plus_rides_by_percentage_thresholds(df, eda_dir):
    """
    Analyze distribution of rides_plus_lifetime for users above different percent_plus_lifetime_rides thresholds.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        eda_dir (Path): Directory to save results
    """
    print("\n=== Analyzing Plus Rides Distribution by Percentage Thresholds ===")
    
    # Check if required columns exist
    if 'rides_plus_lifetime' not in df.columns or 'rides_lifetime' not in df.columns:
        print("Warning: Both rides_plus_lifetime and rides_lifetime are needed for threshold analysis.")
        return
    
    # Calculate percent_plus_lifetime_rides
    df_analysis = df[['rides_plus_lifetime', 'rides_lifetime']].dropna().copy()
    df_analysis['percent_plus_lifetime_rides'] = (
        df_analysis['rides_plus_lifetime'] / df_analysis['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    print(f"Analysis data: {len(df_analysis)} rows")
    
    # Define thresholds (convert percentages to decimals)
    thresholds = [-0.05, 0, 0.05, 0.10, 0.15]  # -5%, 0%, 5%, 10%, 15%
    threshold_labels = ['-5%', '0%', '5%', '10%', '15%']
    
    # Create subplots for histograms
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Plus Rides by Percentage Thresholds', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # Store results for CSV export
    threshold_results = []
    
    for i, (threshold, label) in enumerate(zip(thresholds, threshold_labels)):
        print(f"\nAnalyzing threshold: {label} ({threshold})")
        
        # Filter users above threshold
        filtered_data = df_analysis[df_analysis['percent_plus_lifetime_rides'] > threshold]
        plus_rides = filtered_data['rides_plus_lifetime']
        
        print(f"  Users above {label}: {len(filtered_data):,}")
        if len(filtered_data) > 0:
            print(f"  Plus rides - Mean: {plus_rides.mean():.2f}, Median: {plus_rides.median():.2f}")
            print(f"  Plus rides - Min: {plus_rides.min()}, Max: {plus_rides.max()}")
        
        # Calculate value counts for 0-9, 10-19, ≥20 rides (used for both plot and CSV)
        ride_values = list(range(0, 10)) + [10, 20]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
        counts = []
        
        if len(plus_rides) > 0:
            for ride_val in ride_values:
                if ride_val == 10:
                    # For value 10, include rides 10-19
                    count = ((plus_rides >= 10) & (plus_rides < 20)).sum()
                elif ride_val == 20:
                    # For value 20, include all rides >= 20
                    count = (plus_rides >= 20).sum()
                else:
                    # For values 0-9, exact match
                    count = (plus_rides == ride_val).sum()
                counts.append(count)
        else:
            counts = [0] * len(ride_values)
        
        # Create bar plot
        ax = axes[i]
        if len(plus_rides) > 0:
            # Calculate normalized percentages (adds up to 100%)
            total_users = len(plus_rides)
            percentages = [(count / total_users) * 100 for count in counts]
            
            # Create bar plot with normalized percentages using sequential positions
            x_positions = range(len(ride_values))
            bars = ax.bar(x_positions, percentages, alpha=0.7, color=f'C{i}', edgecolor='black')
            
            # Add value labels on top of bars (show both percentage and count)
            for bar, count, pct in zip(bars, counts, percentages):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{pct:.1f}%\n({count:,})', ha='center', va='bottom', fontsize=7)
            
            # Add statistics text (use original data for stats)
            stats_text = f'n={len(plus_rides):,}\nMean={plus_rides.mean():.1f}\nMedian={plus_rides.median():.1f}'
            pct_above_20 = (plus_rides >= 20).sum() / len(plus_rides) * 100
            if pct_above_20 > 0:
                stats_text += f'\n≥20: {pct_above_20:.1f}%'
            
            ax.text(0.65, 0.8, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=9)
            
            # Set x-axis to show 0-9, 10-19, ≥20 clearly
            ax.set_xlim(-0.5, len(ride_values) - 0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(ride_val) if ride_val < 10 else ('10-19' if ride_val == 10 else '≥20') for ride_val in ride_values])
            
            # Set y-axis to show 0-100%
            ax.set_ylim(0, max(percentages) * 1.1 if percentages else 100)
        else:
            ax.text(0.5, 0.5, 'No data\nabove threshold', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'Plus Rides Distribution\n(percent_plus > {label})')
        ax.set_xlabel('rides_plus_lifetime')
        ax.set_ylabel('Percentage (%)')
        ax.grid(True, alpha=0.3)
        
        # Store results for CSV (using same logic as bar plot)
        if len(filtered_data) > 0:
            total_users = len(plus_rides)
            # Store value counts for 0-9, 10-19, ≥20 rides (same as bar plot)
            for ride_val, count in zip(ride_values, counts):
                if count > 0:  # Only include values that have data
                    pct_normalized = (count / total_users) * 100  # Normalized percentage (adds to 100%)
                    if ride_val == 10:
                        rides_plus_value = '10-19'
                    elif ride_val == 20:
                        rides_plus_value = '20+'
                    else:
                        rides_plus_value = str(ride_val)
                    
                    threshold_results.append({
                        'threshold': label,
                        'threshold_decimal': threshold,
                        'users_above_threshold': len(filtered_data),
                        'rides_plus_value': rides_plus_value,
                        'count': count,
                        'percentage_normalized': pct_normalized,  # % of filtered users with this exact ride count
                        'mean_plus_rides': plus_rides.mean(),
                        'median_plus_rides': plus_rides.median(),
                        'std_plus_rides': plus_rides.std()
                    })
    
    # Hide empty subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = eda_dir / 'plus_rides_distribution_by_thresholds.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved threshold distribution plots to: {plot_file}")
    plt.close()
    
    # Save detailed results to CSV
    if threshold_results:
        results_df = pd.DataFrame(threshold_results)
        results_file = eda_dir / 'plus_rides_distribution_by_thresholds.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Saved threshold analysis results to: {results_file}")
    
    # Create summary statistics table
    summary_stats = []
    for threshold, label in zip(thresholds, threshold_labels):
        filtered_data = df_analysis[df_analysis['percent_plus_lifetime_rides'] > threshold]
        plus_rides = filtered_data['rides_plus_lifetime']
        
        if len(plus_rides) > 0:
            summary_stats.append({
                'threshold': label,
                'threshold_decimal': threshold,
                'users_above_threshold': len(filtered_data),
                'pct_of_all_users': (len(filtered_data) / len(df_analysis)) * 100,
                'mean_plus_rides': plus_rides.mean(),
                'median_plus_rides': plus_rides.median(),
                'std_plus_rides': plus_rides.std(),
                'min_plus_rides': plus_rides.min(),
                'max_plus_rides': plus_rides.max(),
                'q25_plus_rides': plus_rides.quantile(0.25),
                'q75_plus_rides': plus_rides.quantile(0.75)
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = eda_dir / 'plus_rides_threshold_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved threshold summary statistics to: {summary_file}")
        
        # Print summary to console
        print(f"\n=== Threshold Summary ===")
        for _, row in summary_df.iterrows():
            print(f"{row['threshold']}: {row['users_above_threshold']:,} users ({row['pct_of_all_users']:.1f}%), "
                  f"avg {row['mean_plus_rides']:.1f} plus rides")

def analyze_lifetime_rides_by_percentage_thresholds(df, eda_dir):
    """
    Analyze distribution of rides_lifetime for users above different percent_plus_lifetime_rides thresholds.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        eda_dir (Path): Directory to save results
    """
    print("\n=== Analyzing Lifetime Rides Distribution by Percentage Thresholds ===")
    
    # Check if required columns exist
    if 'rides_plus_lifetime' not in df.columns or 'rides_lifetime' not in df.columns:
        print("Warning: Both rides_plus_lifetime and rides_lifetime are needed for threshold analysis.")
        return
    
    # Calculate percent_plus_lifetime_rides
    df_analysis = df[['rides_plus_lifetime', 'rides_lifetime']].dropna().copy()
    df_analysis['percent_plus_lifetime_rides'] = (
        df_analysis['rides_plus_lifetime'] / df_analysis['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    print(f"Analysis data: {len(df_analysis)} rows")
    
    # Define thresholds (convert percentages to decimals)
    thresholds = [-0.05, 0, 0.05, 0.10, 0.15]  # -5%, 0%, 5%, 10%, 15%
    threshold_labels = ['-5%', '0%', '5%', '10%', '15%']
    
    # Create subplots for histograms
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Lifetime Rides by Percentage Thresholds', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # Store results for CSV export
    threshold_results = []
    
    for i, (threshold, label) in enumerate(zip(thresholds, threshold_labels)):
        print(f"\nAnalyzing threshold: {label} ({threshold})")
        
        # Filter users above threshold
        filtered_data = df_analysis[df_analysis['percent_plus_lifetime_rides'] > threshold]
        lifetime_rides = filtered_data['rides_lifetime']
        
        print(f"  Users above {label}: {len(filtered_data):,}")
        if len(filtered_data) > 0:
            print(f"  Lifetime rides - Mean: {lifetime_rides.mean():.2f}, Median: {lifetime_rides.median():.2f}")
            print(f"  Lifetime rides - Min: {lifetime_rides.min()}, Max: {lifetime_rides.max()}")
        
        # Calculate value counts for 0-9, 10-19, ≥20 rides (used for both plot and CSV)
        ride_values = list(range(0, 10)) + [10, 20]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
        counts = []
        
        if len(lifetime_rides) > 0:
            for ride_val in ride_values:
                if ride_val == 10:
                    # For value 10, include rides 10-19
                    count = ((lifetime_rides >= 10) & (lifetime_rides < 20)).sum()
                elif ride_val == 20:
                    # For value 20, include all rides >= 20
                    count = (lifetime_rides >= 20).sum()
                else:
                    # For values 0-9, exact match
                    count = (lifetime_rides == ride_val).sum()
                counts.append(count)
        else:
            counts = [0] * len(ride_values)
        
        # Create bar plot
        ax = axes[i]
        if len(lifetime_rides) > 0:
            # Calculate normalized percentages (adds up to 100%)
            total_users = len(lifetime_rides)
            percentages = [(count / total_users) * 100 for count in counts]
            
            # Create bar plot with normalized percentages using sequential positions
            x_positions = range(len(ride_values))
            bars = ax.bar(x_positions, percentages, alpha=0.7, color=f'C{i}', edgecolor='black')
            
            # Add value labels on top of bars (show both percentage and count)
            for bar, count, pct in zip(bars, counts, percentages):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{pct:.1f}%\n({count:,})', ha='center', va='bottom', fontsize=7)
            
            # Add statistics text (use original data for stats)
            stats_text = f'n={len(lifetime_rides):,}\nMean={lifetime_rides.mean():.1f}\nMedian={lifetime_rides.median():.1f}'
            pct_above_20 = (lifetime_rides >= 20).sum() / len(lifetime_rides) * 100
            if pct_above_20 > 0:
                stats_text += f'\n≥20: {pct_above_20:.1f}%'
            
            ax.text(0.65, 0.8, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=9)
            
            # Set x-axis to show 0-9, 10-19, ≥20 clearly
            ax.set_xlim(-0.5, len(ride_values) - 0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(ride_val) if ride_val < 10 else ('10-19' if ride_val == 10 else '≥20') for ride_val in ride_values])
            
            # Set y-axis to show 0-100%
            ax.set_ylim(0, max(percentages) * 1.1 if percentages else 100)
        else:
            ax.text(0.5, 0.5, 'No data\nabove threshold', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'Lifetime Rides Distribution\n(percent_plus > {label})')
        ax.set_xlabel('rides_lifetime')
        ax.set_ylabel('Percentage (%)')
        ax.grid(True, alpha=0.3)
        
        # Store results for CSV (using same logic as bar plot)
        if len(filtered_data) > 0:
            total_users = len(lifetime_rides)
            # Store value counts for 0-9, 10-19, ≥20 rides (same as bar plot)
            for ride_val, count in zip(ride_values, counts):
                if count > 0:  # Only include values that have data
                    pct_normalized = (count / total_users) * 100  # Normalized percentage (adds to 100%)
                    if ride_val == 10:
                        rides_lifetime_value = '10-19'
                    elif ride_val == 20:
                        rides_lifetime_value = '20+'
                    else:
                        rides_lifetime_value = str(ride_val)
                    
                    threshold_results.append({
                        'threshold': label,
                        'threshold_decimal': threshold,
                        'users_above_threshold': len(filtered_data),
                        'rides_lifetime_value': rides_lifetime_value,
                        'count': count,
                        'percentage_normalized': pct_normalized,  # % of filtered users with this exact ride count
                        'mean_lifetime_rides': lifetime_rides.mean(),
                        'median_lifetime_rides': lifetime_rides.median(),
                        'std_lifetime_rides': lifetime_rides.std()
                    })
    
    # Hide empty subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = eda_dir / 'lifetime_rides_distribution_by_thresholds.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved threshold distribution plots to: {plot_file}")
    plt.close()
    
    # Save detailed results to CSV
    if threshold_results:
        results_df = pd.DataFrame(threshold_results)
        results_file = eda_dir / 'lifetime_rides_distribution_by_thresholds.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Saved threshold analysis results to: {results_file}")
    
    # Create summary statistics table
    summary_stats = []
    for threshold, label in zip(thresholds, threshold_labels):
        filtered_data = df_analysis[df_analysis['percent_plus_lifetime_rides'] > threshold]
        lifetime_rides = filtered_data['rides_lifetime']
        
        if len(lifetime_rides) > 0:
            summary_stats.append({
                'threshold': label,
                'threshold_decimal': threshold,
                'users_above_threshold': len(filtered_data),
                'pct_of_all_users': (len(filtered_data) / len(df_analysis)) * 100,
                'mean_lifetime_rides': lifetime_rides.mean(),
                'median_lifetime_rides': lifetime_rides.median(),
                'std_lifetime_rides': lifetime_rides.std(),
                'min_lifetime_rides': lifetime_rides.min(),
                'max_lifetime_rides': lifetime_rides.max(),
                'q25_lifetime_rides': lifetime_rides.quantile(0.25),
                'q75_lifetime_rides': lifetime_rides.quantile(0.75)
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = eda_dir / 'lifetime_rides_threshold_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved threshold summary statistics to: {summary_file}")
        
        # Print summary to console
        print(f"\n=== Threshold Summary ===")
        for _, row in summary_df.iterrows():
            print(f"{row['threshold']}: {row['users_above_threshold']:,} users ({row['pct_of_all_users']:.1f}%), "
                  f"avg {row['mean_lifetime_rides']:.1f} lifetime rides")

def analyze_signup_year_by_percentage_thresholds(df, eda_dir):
    """
    Analyze distribution of signup_year for users above different percent_plus_lifetime_rides thresholds.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        eda_dir (Path): Directory to save results
    """
    print("\n=== Analyzing Signup Year Distribution by Percentage Thresholds ===")
    
    # Check if required columns exist
    if 'rides_plus_lifetime' not in df.columns or 'rides_lifetime' not in df.columns or 'signup_year' not in df.columns:
        print("Warning: rides_plus_lifetime, rides_lifetime, and signup_year are all needed for threshold analysis.")
        return
    
    # Calculate percent_plus_lifetime_rides
    df_analysis = df[['rides_plus_lifetime', 'rides_lifetime', 'signup_year']].dropna().copy()
    df_analysis['percent_plus_lifetime_rides'] = (
        df_analysis['rides_plus_lifetime'] / df_analysis['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    print(f"Analysis data: {len(df_analysis)} rows")
    
    # Get unique signup years and sort them
    signup_years = sorted(df_analysis['signup_year'].unique())
    print(f"Signup years available: {signup_years}")
    
    # Define thresholds (convert percentages to decimals)
    thresholds = [-0.05, 0, 0.05, 0.10, 0.15]  # -5%, 0%, 5%, 10%, 15%
    threshold_labels = ['-5%', '0%', '5%', '10%', '15%']
    
    # Create subplots for histograms
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Distribution of Signup Year by Percentage Thresholds', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # Store results for CSV export
    threshold_results = []
    
    for i, (threshold, label) in enumerate(zip(thresholds, threshold_labels)):
        print(f"\nAnalyzing threshold: {label} ({threshold})")
        
        # Filter users above threshold
        filtered_data = df_analysis[df_analysis['percent_plus_lifetime_rides'] > threshold]
        signup_year_data = filtered_data['signup_year']
        
        print(f"  Users above {label}: {len(filtered_data):,}")
        if len(filtered_data) > 0:
            print(f"  Signup years - Min: {signup_year_data.min()}, Max: {signup_year_data.max()}")
            print(f"  Signup years - Mode: {signup_year_data.mode().iloc[0] if not signup_year_data.mode().empty else 'N/A'}")
        
        # Calculate value counts for each signup year
        year_counts = []
        if len(signup_year_data) > 0:
            for year in signup_years:
                count = (signup_year_data == year).sum()
                year_counts.append(count)
        else:
            year_counts = [0] * len(signup_years)
        
        # Create bar plot
        ax = axes[i]
        if len(signup_year_data) > 0:
            # Calculate normalized percentages (adds up to 100%)
            total_users = len(signup_year_data)
            percentages = [(count / total_users) * 100 for count in year_counts]
            
            # Create bar plot with normalized percentages using sequential positions
            x_positions = range(len(signup_years))
            bars = ax.bar(x_positions, percentages, alpha=0.7, color=f'C{i}', edgecolor='black')
            
            # Add value labels on top of bars (show both percentage and count)
            for bar, count, pct in zip(bars, year_counts, percentages):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{pct:.1f}%\n({count:,})', ha='center', va='bottom', fontsize=7)
            
            # Add statistics text (use original data for stats)
            stats_text = f'n={len(signup_year_data):,}\nMin={signup_year_data.min()}\nMax={signup_year_data.max()}'
            if not signup_year_data.mode().empty:
                stats_text += f'\nMode={signup_year_data.mode().iloc[0]}'
            
            ax.text(0.65, 0.8, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=9)
            
            # Set x-axis to show signup years clearly
            ax.set_xlim(-0.5, len(signup_years) - 0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(year) for year in signup_years], rotation=45)
            
            # Set y-axis to show 0-100%
            ax.set_ylim(0, max(percentages) * 1.1 if percentages else 100)
        else:
            ax.text(0.5, 0.5, 'No data\nabove threshold', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_title(f'Signup Year Distribution\n(percent_plus > {label})')
        ax.set_xlabel('signup_year')
        ax.set_ylabel('Percentage (%)')
        ax.grid(True, alpha=0.3)
        
        # Store results for CSV (using same logic as bar plot)
        if len(filtered_data) > 0:
            total_users = len(signup_year_data)
            # Store value counts for each signup year
            for year, count in zip(signup_years, year_counts):
                if count > 0:  # Only include years that have data
                    pct_normalized = (count / total_users) * 100  # Normalized percentage (adds to 100%)
                    threshold_results.append({
                        'threshold': label,
                        'threshold_decimal': threshold,
                        'users_above_threshold': len(filtered_data),
                        'signup_year': year,
                        'count': count,
                        'percentage_normalized': pct_normalized,  # % of filtered users with this signup year
                        'mean_plus_rides': filtered_data['rides_plus_lifetime'].mean(),
                        'median_plus_rides': filtered_data['rides_plus_lifetime'].median(),
                        'std_plus_rides': filtered_data['rides_plus_lifetime'].std()
                    })
    
    # Hide empty subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = eda_dir / 'signup_year_distribution_by_thresholds.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved threshold distribution plots to: {plot_file}")
    plt.close()
    
    # Save detailed results to CSV
    if threshold_results:
        results_df = pd.DataFrame(threshold_results)
        results_file = eda_dir / 'signup_year_distribution_by_thresholds.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Saved threshold analysis results to: {results_file}")
    
    # Create summary statistics table
    summary_stats = []
    for threshold, label in zip(thresholds, threshold_labels):
        filtered_data = df_analysis[df_analysis['percent_plus_lifetime_rides'] > threshold]
        signup_year_data = filtered_data['signup_year']
        
        if len(signup_year_data) > 0:
            summary_stats.append({
                'threshold': label,
                'threshold_decimal': threshold,
                'users_above_threshold': len(filtered_data),
                'pct_of_all_users': (len(filtered_data) / len(df_analysis)) * 100,
                'min_signup_year': signup_year_data.min(),
                'max_signup_year': signup_year_data.max(),
                'mode_signup_year': signup_year_data.mode().iloc[0] if not signup_year_data.mode().empty else None,
                'mean_signup_year': signup_year_data.mean(),
                'median_signup_year': signup_year_data.median(),
                'std_signup_year': signup_year_data.std()
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_file = eda_dir / 'signup_year_threshold_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved threshold summary statistics to: {summary_file}")
        
        # Print summary to console
        print(f"\n=== Threshold Summary ===")
        for _, row in summary_df.iterrows():
            print(f"{row['threshold']}: {row['users_above_threshold']:,} users ({row['pct_of_all_users']:.1f}%), "
                  f"years {row['min_signup_year']}-{row['max_signup_year']}, mode: {row['mode_signup_year']}")

def create_comparison_plots(df, eda_dir):
    """
    Create comparison plots between rides_plus_lifetime and rides_lifetime.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        eda_dir (Path): Directory to save plots
    """
    print("\n=== Creating Comparison Plots ===")
    
    # Check if both features exist
    if 'rides_plus_lifetime' not in df.columns or 'rides_lifetime' not in df.columns:
        print("Warning: Both rides_plus_lifetime and rides_lifetime are needed for comparison.")
        return
    
    # Get data where both features are non-null
    comparison_data = df[['rides_plus_lifetime', 'rides_lifetime']].dropna()
    
    if len(comparison_data) == 0:
        print("Warning: No rows with both features non-null.")
        return
    
    print(f"Comparison data: {len(comparison_data)} rows")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rides Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(comparison_data['rides_lifetime'], comparison_data['rides_plus_lifetime'], 
               alpha=0.5, s=10, color='blue')
    ax1.set_xlabel('rides_lifetime')
    ax1.set_ylabel('rides_plus_lifetime')
    ax1.set_title('Scatter Plot: Plus vs Total Rides')
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal line (y=x)
    max_val = max(comparison_data['rides_lifetime'].max(), comparison_data['rides_plus_lifetime'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='y=x')
    ax1.legend()
    
    # 2. Plus percentage distribution
    ax2 = axes[0, 1]
    plus_pct = (comparison_data['rides_plus_lifetime'] / comparison_data['rides_lifetime'].replace(0, np.nan)).fillna(0)
    plus_pct = plus_pct.clip(0, 1)  # Clip to 0-1 range
    
    ax2.hist(plus_pct, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Plus Percentage (rides_plus_lifetime / rides_lifetime)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Plus Percentage')
    ax2.grid(True, alpha=0.3)
    
    # 3. Side-by-side box plots
    ax3 = axes[1, 0]
    box_data = [comparison_data['rides_lifetime'], comparison_data['rides_plus_lifetime']]
    box_labels = ['rides_lifetime', 'rides_plus_lifetime']
    ax3.boxplot(box_data, labels=box_labels)
    ax3.set_title('Box Plot Comparison')
    ax3.set_ylabel('Number of Rides')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation and statistics
    ax4 = axes[1, 1]
    correlation = comparison_data['rides_lifetime'].corr(comparison_data['rides_plus_lifetime'])
    
    # Create a text summary
    summary_text = f"""
    Correlation: {correlation:.3f}
    
    rides_lifetime:
    Mean: {comparison_data['rides_lifetime'].mean():.2f}
    Std: {comparison_data['rides_lifetime'].std():.2f}
    Median: {comparison_data['rides_lifetime'].median():.2f}
    
    rides_plus_lifetime:
    Mean: {comparison_data['rides_plus_lifetime'].mean():.2f}
    Std: {comparison_data['rides_plus_lifetime'].std():.2f}
    Median: {comparison_data['rides_plus_lifetime'].median():.2f}
    
    Plus Percentage:
    Mean: {plus_pct.mean():.2f}
    Std: {plus_pct.std():.2f}
    Median: {plus_pct.median():.2f}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10, fontfamily='monospace')
    ax4.set_title('Summary Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = eda_dir / 'rides_comparison_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plots to: {plot_file}")
    plt.close()
    
    # Save comparison statistics to CSV
    comparison_stats = {
        'metric': ['correlation', 'rides_lifetime_mean', 'rides_lifetime_std', 'rides_lifetime_median',
                  'rides_plus_lifetime_mean', 'rides_plus_lifetime_std', 'rides_plus_lifetime_median',
                  'plus_percentage_mean', 'plus_percentage_std', 'plus_percentage_median'],
        'value': [correlation, comparison_data['rides_lifetime'].mean(), comparison_data['rides_lifetime'].std(),
                 comparison_data['rides_lifetime'].median(), comparison_data['rides_plus_lifetime'].mean(),
                 comparison_data['rides_plus_lifetime'].std(), comparison_data['rides_plus_lifetime'].median(),
                 plus_pct.mean(), plus_pct.std(), plus_pct.median()]
    }
    
    comparison_stats_df = pd.DataFrame(comparison_stats)
    comparison_stats_file = eda_dir / 'rides_comparison_stats.csv'
    comparison_stats_df.to_csv(comparison_stats_file, index=False)
    print(f"Saved comparison statistics to: {comparison_stats_file}")

def main():
    """Main function to run the EDA analysis."""
    print("Starting EDA Analysis: Rides Distribution")
    print("=" * 50)
    
    # Create EDA directory
    eda_dir = create_eda_directory()
    print(f"EDA directory: {eda_dir}")
    
    # Load data
    df = load_and_prepare_data()
    
    # Features to analyze
    features = ['rides_plus_lifetime', 'rides_lifetime']
    
    # Analyze each feature
    for feature in features:
        stats, feature_data = analyze_feature_distribution(df, feature, eda_dir)
        if stats is not None and feature_data is not None:
            create_distribution_plots(df, feature, feature_data, eda_dir)
    
    # Analyze plus rides distribution by percentage thresholds
    analyze_plus_rides_by_percentage_thresholds(df, eda_dir)
    
    # Analyze lifetime rides distribution by percentage thresholds
    analyze_lifetime_rides_by_percentage_thresholds(df, eda_dir)

    # Analyze signup year distribution by percentage thresholds
    analyze_signup_year_by_percentage_thresholds(df, eda_dir)
    
    # Create comparison plots
    create_comparison_plots(df, eda_dir)
    
    print(f"\n{'='*50}")
    print("EDA Analysis Complete!")
    print(f"Results saved to: {eda_dir}")
    print("Files created:")
    print("  - rides_plus_lifetime_descriptive_stats.csv")
    print("  - rides_plus_lifetime_value_counts.csv")
    print("  - rides_plus_lifetime_distribution_plots.png")
    print("  - rides_lifetime_descriptive_stats.csv")
    print("  - rides_lifetime_value_counts.csv")
    print("  - rides_lifetime_distribution_plots.png")
    print("  - plus_rides_distribution_by_thresholds.png (normalized percentages)")
    print("  - plus_rides_distribution_by_thresholds.csv (with normalized percentages)")
    print("  - plus_rides_threshold_summary.csv")
    print("  - lifetime_rides_distribution_by_thresholds.png (normalized percentages)")
    print("  - lifetime_rides_distribution_by_thresholds.csv (with normalized percentages)")
    print("  - lifetime_rides_threshold_summary.csv")
    print("  - signup_year_distribution_by_thresholds.png (normalized percentages)")
    print("  - signup_year_distribution_by_thresholds.csv (with normalized percentages)")
    print("  - signup_year_threshold_summary.csv")
    print("  - rides_comparison_plots.png")
    print("  - rides_comparison_stats.csv")

if __name__ == "__main__":
    main() 