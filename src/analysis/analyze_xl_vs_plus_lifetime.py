import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.load_data import load_parquet_data
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def add_churned_indicator(df):
    df['ds'] = pd.to_datetime(df['ds'])
    churned_mask = (
        (df['days_since_signup'] > 365) &
        (df['rides_lifetime'] > 2) &
        (df['all_type_total_rides_365d'] == 0)
    )
    df['is_churned_user'] = churned_mask.astype(int)
    return df

def filter_by_segment(df, segment_type):
    """
    Filter dataframe by segment type.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        segment_type (str): Type of segment to filter by
            - 'airport': Sessions where destination_venue_category = 'airport' or origin_venue_category = 'airport'
            - 'airport_dropoff': Sessions where destination_venue_category = 'airport'
            - 'airport_pickup': Sessions where origin_venue_category = 'airport'
            - 'churned': Sessions where rider is churned (is_churned_user = 1)
            - 'all': No filtering (use all data)
    
    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    if segment_type == 'airport':
        airport_mask = (
            (df['destination_venue_category'] == 'airport') |
            (df['origin_venue_category'] == 'airport')
        )
        filtered_df = df[airport_mask].copy()
        print(f"Airport sessions: {len(filtered_df)} rows (from {len(df)} total)")
    elif segment_type == 'airport_dropoff':
        airport_dropoff_mask = (df['destination_venue_category'] == 'airport')
        filtered_df = df[airport_dropoff_mask].copy()
        print(f"Airport dropoff sessions: {len(filtered_df)} rows (from {len(df)} total)")
    elif segment_type == 'airport_pickup':
        airport_pickup_mask = (df['origin_venue_category'] == 'airport')
        filtered_df = df[airport_pickup_mask].copy()
        print(f"Airport pickup sessions: {len(filtered_df)} rows (from {len(df)} total)")
    elif segment_type == 'churned':
        churned_mask = (df['is_churned_user'] == 1)
        filtered_df = df[churned_mask].copy()
        print(f"Churned rider sessions: {len(filtered_df)} rows (from {len(df)} total)")
    elif segment_type == 'all':
        filtered_df = df.copy()
        print(f"Using all data: {len(filtered_df)} rows")
    else:
        raise ValueError(f"Unknown segment type: {segment_type}. Use 'airport', 'airport_dropoff', 'airport_pickup', 'churned', or 'all'")
    return filtered_df

def load_and_prepare_data(segment_type='all', data_version='v2'):
    """Load and prepare data for Plus vs standard_saver choice analysis."""
    print(f"Loading and preparing data for segment: {segment_type} (data_version={data_version})...")
    df = load_parquet_data(data_version)
    df = add_churned_indicator(df)
    # Filter by segment
    df = filter_by_segment(df, segment_type)
    # Basic filtering
    required_cols = ['requested_ride_type', 'rides_lifetime', 'rides_plus_lifetime', 'preselected_mode', 
                     'plus_availability_caveat', 'standard_saver_availability_caveat']
    df = df.dropna(subset=required_cols)
    print(f"Data loaded: {len(df)} rows")
    return df

def calculate_plus_percentage(df):
    """Calculate percentage of lifetime rides that were Plus."""
    # Handle division by zero by replacing 0 with NaN, then filling with 0
    df['percent_rides_plus_lifetime'] = (
        df['rides_plus_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    # Clip to 0-1 range (in case of data issues)
    #df['percent_rides_plus_lifetime'] = df['percent_rides_plus_lifetime'].clip(0, 1)
    
    return df

def create_plus_target(df):
    """Create binary target for Plus requests."""
    # Filter out sessions where Plus was preselected
    df = df[df['preselected_mode'] != 'plus'].copy()
    
    df['chose_plus'] = (df['requested_ride_type'] == 'plus').astype(int)
    return df

def analyze_plus_vs_plus_percentage(df, total_sessions_all=None, total_riders_all=None):
    """Analyze the relationship between Plus choice and Plus percentage."""
    print("\n=== Plus vs Plus Lifetime Analysis ===")
    
    # Overall statistics
    total_sessions = len(df)
    plus_sessions = df['chose_plus'].sum()
    plus_rate = plus_sessions / total_sessions
    
    print(f"Total sessions: {total_sessions:,}")
    print(f"Plus sessions: {plus_sessions:,}")
    print(f"Overall Plus rate: {plus_rate:.3f} ({plus_rate*100:.1f}%)")
    
    # Plus percentage statistics
    plus_pct_stats = df['percent_rides_plus_lifetime'].describe()
    print(f"\nPlus percentage statistics:")
    print(plus_pct_stats)
    
    # Analyze by Plus percentage bins
    print(f"\n=== Analysis by Plus Percentage Bins ===")
    
    # Create bins for Plus percentage
    df['plus_pct_bin'] = pd.cut(
        df['percent_rides_plus_lifetime'], 
        bins=[0, 0.1, 0.25, 0.5, 0.75, 1.0], 
        labels=['0-10%', '10-25%', '25-50%', '50-75%', '75-100%'],
        include_lowest=True
    )
    
    bin_analysis = df.groupby('plus_pct_bin').agg({
        'chose_plus': ['count', 'sum', 'mean'],
        'percent_rides_plus_lifetime': 'mean'
    }).round(4)
    
    bin_analysis.columns = ['total_sessions', 'plus_sessions', 'plus_rate', 'avg_plus_pct']
    bin_analysis['plus_rate_pct'] = bin_analysis['plus_rate'] * 100
    
    print(bin_analysis)
    
    # Create threshold analysis table
    print(f"\n=== Threshold Analysis Table ===")
    threshold_analysis = create_threshold_table(df, total_sessions_all, total_riders_all)
    print(threshold_analysis)
    
    return df, bin_analysis, threshold_analysis

def create_threshold_table(df, total_sessions_all=None, total_riders_all=None):
    """Create a table showing probability of choosing Plus and standard_saver for sessions where percent_rides_plus_lifetime > threshold."""
    # Define thresholds
    thresholds = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Use provided totals or calculate from current dataset
    if total_sessions_all is None:
        total_sessions_all = len(df)
    if total_riders_all is None:
        total_riders_all = df['rider_lyft_id'].nunique()
    
    results = []
    for threshold in thresholds:
        # Filter for sessions where percent_rides_plus_lifetime > threshold
        mask = df['percent_rides_plus_lifetime'] > threshold
        subset = df[mask]
        
        if len(subset) > 0:
            plus_rate = subset['chose_plus'].mean()
            plus_sessions = subset['chose_plus'].sum()
            saver_rate = (subset['requested_ride_type'] == 'standard_saver').mean()
            saver_sessions = (subset['requested_ride_type'] == 'standard_saver').sum()
            total_sessions = len(subset)
            distinct_riders = subset['rider_lyft_id'].nunique()
            
            # Calculate percentages normalized by the whole dataset totals
            sessions_pct = (total_sessions / total_sessions_all) * 100
            riders_pct = (distinct_riders / total_riders_all) * 100
            
            results.append({
                'threshold': threshold,
                'threshold_pct': f"{threshold*100:.0f}%",
                'total_sessions': total_sessions,
                'sessions_pct': f"{sessions_pct:.1f}%",
                'distinct_riders': distinct_riders,
                'riders_pct': f"{riders_pct:.1f}%",
                'plus_sessions': plus_sessions,
                'plus_probability': plus_rate,
                'plus_probability_pct': f"{plus_rate*100:.1f}%",
                'saver_sessions': saver_sessions,
                'saver_probability': saver_rate,
                'saver_probability_pct': f"{saver_rate*100:.1f}%"
            })
    
    threshold_df = pd.DataFrame(results)
    return threshold_df

def create_visualizations(df, bin_analysis, output_dir):
    """Create visualizations for the analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution of Plus percentage
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['percent_rides_plus_lifetime'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Percent Rides Plus Lifetime')
    plt.ylabel('Frequency')
    plt.title('Distribution of Plus Percentage')
    plt.grid(True, alpha=0.3)
    
    # 2. Plus rate by Plus percentage bins
    plt.subplot(2, 2, 2)
    x_pos = range(len(bin_analysis))
    plt.bar(x_pos, bin_analysis['plus_rate_pct'], alpha=0.7, edgecolor='black')
    plt.xlabel('Plus Percentage Bin')
    plt.ylabel('Plus Rate (%)')
    plt.title('Plus Rate by Plus Percentage')
    plt.xticks(x_pos, bin_analysis.index, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(bin_analysis['plus_rate_pct']):
        plt.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. Scatter plot with trend line
    plt.subplot(2, 2, 3)
    # Sample data for scatter plot (too many points otherwise)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    plt.scatter(df_sample['percent_rides_plus_lifetime'], df_sample['chose_plus'], 
               alpha=0.1, s=1)
    
    # Add trend line using binned averages
    plt.plot(bin_analysis['avg_plus_pct'], bin_analysis['plus_rate_pct'], 
             'ro-', linewidth=2, markersize=8, label='Binned Average')
    
    plt.xlabel('Percent Rides Plus Lifetime')
    plt.ylabel('Plus Rate (%)')
    plt.title('Plus Rate vs Plus Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Box plot of Plus percentage by Plus choice
    plt.subplot(2, 2, 4)
    plus_data = df[df['chose_plus'] == 1]['percent_rides_plus_lifetime']
    non_plus_data = df[df['chose_plus'] == 0]['percent_rides_plus_lifetime']
    
    plt.boxplot([non_plus_data, plus_data], labels=['Non-Plus', 'Plus'])
    plt.ylabel('Percent Rides Plus Lifetime')
    plt.title('Plus Percentage Distribution by Plus Choice')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plus_vs_plus_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plus_vs_plus_analysis.pdf', bbox_inches='tight')
    print(f"Saved visualizations to {output_dir}")
    
    # Create detailed correlation analysis
    create_correlation_analysis(df, output_dir)

def create_correlation_analysis(df, output_dir):
    """Create detailed correlation analysis."""
    # Calculate correlation
    correlation = df['percent_rides_plus_lifetime'].corr(df['chose_plus'])
    
    # Create correlation plot
    plt.figure(figsize=(10, 6))
    
    # Scatter with jitter for binary variable
    df_sample = df.sample(n=min(5000, len(df)), random_state=42)
    plt.scatter(df_sample['percent_rides_plus_lifetime'], 
               df_sample['chose_plus'] + np.random.normal(0, 0.02, len(df_sample)), 
               alpha=0.3, s=1)
    
    # Add trend line
    z = np.polyfit(df['percent_rides_plus_lifetime'], df['chose_plus'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(0, 1, 100)
    plt.plot(x_range, p(x_range), "r--", linewidth=2, 
             label=f'Trend line (correlation: {correlation:.3f})')
    
    plt.xlabel('Percent Rides Plus Lifetime')
    plt.ylabel('Chose Plus (with jitter)')
    plt.title(f'Correlation Analysis: Plus Choice vs Plus Percentage\nCorrelation: {correlation:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'plus_plus_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'plus_plus_correlation.pdf', bbox_inches='tight')
    
    print(f"Correlation between Plus choice and Plus percentage: {correlation:.3f}")

def create_summary_report(df, bin_analysis, threshold_analysis, output_dir):
    """Create a summary report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    report_path = output_dir / 'plus_vs_plus_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("Plus vs Plus Lifetime Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        total_sessions = len(df)
        total_riders = df['rider_lyft_id'].nunique()
        plus_sessions = df['chose_plus'].sum()
        plus_rate = plus_sessions / total_sessions
        
        f.write(f"Overall Statistics:\n")
        f.write(f"- Total sessions: {total_sessions:,}\n")
        f.write(f"- Total distinct riders: {total_riders:,}\n")
        f.write(f"- Plus sessions: {plus_sessions:,}\n")
        f.write(f"- Overall Plus rate: {plus_rate:.3f} ({plus_rate*100:.1f}%)\n\n")
        
        # Plus percentage statistics
        f.write(f"Plus Percentage Statistics:\n")
        f.write(f"{df['percent_rides_plus_lifetime'].describe().to_string()}\n\n")
        
        # Bin analysis
        f.write(f"Plus Rate by Plus Percentage Bins:\n")
        f.write(f"{bin_analysis.to_string()}\n\n")
        
        # Correlation
        correlation = df['percent_rides_plus_lifetime'].corr(df['chose_plus'])
        f.write(f"Correlation Analysis:\n")
        f.write(f"- Correlation coefficient: {correlation:.3f}\n")
        f.write(f"- Interpretation: {'Positive' if correlation > 0 else 'Negative'} correlation\n")
        f.write(f"- Strength: {'Strong' if abs(correlation) > 0.5 else 'Moderate' if abs(correlation) > 0.3 else 'Weak'}\n\n")
        
        # Key insights
        f.write(f"Key Insights:\n")
        max_plus_bin = bin_analysis.loc[bin_analysis['plus_rate'].idxmax()]
        min_plus_bin = bin_analysis.loc[bin_analysis['plus_rate'].idxmin()]
        
        f.write(f"- Highest Plus rate: {max_plus_bin.name} ({max_plus_bin['plus_rate_pct']:.1f}%)\n")
        f.write(f"- Lowest Plus rate: {min_plus_bin.name} ({min_plus_bin['plus_rate_pct']:.1f}%)\n")
        f.write(f"- Range: {max_plus_bin['plus_rate_pct'] - min_plus_bin['plus_rate_pct']:.1f} percentage points\n\n")
        
        # Threshold analysis
        f.write(f"Threshold Analysis:\n")
        f.write(f"Probability of choosing Plus for sessions where percent_rides_plus_lifetime > threshold:\n")
        f.write(f"{threshold_analysis.to_string(index=False)}\n\n")
        
        # Key threshold insights
        f.write(f"Threshold Insights:\n")
        if len(threshold_analysis) > 1:
            # Find threshold with highest probability
            max_prob_row = threshold_analysis.loc[threshold_analysis['plus_probability'].idxmax()]
            f.write(f"- Highest Plus probability: {max_prob_row['threshold_pct']} threshold ({max_prob_row['plus_probability_pct']})\n")
            f.write(f"- Sessions above this threshold: {max_prob_row['total_sessions']:,}\n")
            f.write(f"- Distinct riders above this threshold: {max_prob_row['distinct_riders']:,}\n")
            
            # Find meaningful threshold (e.g., >20% with reasonable sample size)
            meaningful_thresholds = threshold_analysis[threshold_analysis['total_sessions'] >= 1000]
            if len(meaningful_thresholds) > 0:
                best_threshold = meaningful_thresholds.loc[meaningful_thresholds['plus_probability'].idxmax()]
                f.write(f"- Best threshold with 1000+ sessions: {best_threshold['threshold_pct']} ({best_threshold['plus_probability_pct']})\n")
                f.write(f"- Distinct riders at best threshold: {best_threshold['distinct_riders']:,}\n")
            
            # Find threshold with most riders
            max_riders_row = threshold_analysis.loc[threshold_analysis['distinct_riders'].idxmax()]
            f.write(f"- Threshold with most distinct riders: {max_riders_row['threshold_pct']} ({max_riders_row['distinct_riders']:,} riders)\n")
    
    print(f"Saved summary report to {report_path}")

def main(segment_type_list=['all'], data_version='v2'):
    """Main analysis function."""
    print("Starting Plus vs Plus Lifetime Analysis")
    print(f"Segments: {segment_type_list}")
    print(f"Using {data_version.upper()} data")
    print("=" * 50)
    # Load data once
    print("Loading data...")
    df = load_parquet_data(data_version)
    df = add_churned_indicator(df)
    # Calculate totals for the whole dataset (for normalization)
    print("Calculating whole dataset totals for normalization...")
    df_whole = filter_by_segment(df, 'all')
    # Basic filtering for whole dataset
    required_cols = ['requested_ride_type', 'rides_lifetime', 'rides_plus_lifetime', 'preselected_mode', 
                     'plus_availability_caveat', 'standard_saver_availability_caveat']
    df_whole = df_whole.dropna(subset=required_cols)
    df_whole = calculate_plus_percentage(df_whole)
    df_whole = create_plus_target(df_whole)
    total_sessions_all = len(df_whole)
    total_riders_all = df_whole['rider_lyft_id'].nunique()
    print(f"Whole dataset totals: {total_sessions_all:,} sessions, {total_riders_all:,} riders")
    # Process each segment type
    for segment_type in segment_type_list:
        print(f"\n{'='*60}")
        print(f"Processing segment: {segment_type}")
        print(f"{'='*60}")
        # Filter data for this segment
        df_segment = filter_by_segment(df, segment_type)
        # Basic filtering for segment
        df_segment = df_segment.dropna(subset=required_cols)
        df_segment = calculate_plus_percentage(df_segment)
        df_segment = create_plus_target(df_segment)
        # Analyze relationship
        df_segment, bin_analysis, threshold_analysis = analyze_plus_vs_plus_percentage(df_segment, total_sessions_all, total_riders_all)
        # Create segment-specific output directories
        base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
        plots_dir = base_path / 'plots'
        reports_dir = base_path / f'reports/plus_vs_plus_analysis/segment_{segment_type}'
        # Create visualizations
        create_visualizations(df_segment, bin_analysis, plots_dir)
        # Create summary report
        create_summary_report(df_segment, bin_analysis, threshold_analysis, reports_dir)
        # Save threshold analysis as CSV
        threshold_analysis.to_csv(reports_dir / f'plus_vs_plus_threshold_analysis_{segment_type}.csv', index=False)
        print(f"\nAnalysis complete for segment: {segment_type}")
        print(f"Results saved to:")
        print(f"  - Plots: {plots_dir}")
        print(f"  - Reports: {reports_dir}")
        # Print key findings
        correlation = df_segment['percent_rides_plus_lifetime'].corr(df_segment['chose_plus'])
        print(f"Key Finding: Correlation between Plus choice and Plus percentage: {correlation:.3f}")

        # --- Additional analysis by airline and signup_year ---
        for feature in ['airline', 'signup_year']:
            if feature in df_segment.columns:
                print(f"\n=== Analysis by {feature} ===")
                group = df_segment.groupby(feature).agg(
                    plus_rate=('chose_plus', 'mean'),
                    session_count=('chose_plus', 'count'),
                    unique_riders=('rider_lyft_id', 'nunique')
                ).sort_values('plus_rate', ascending=False)
                print(group.head(10))
                # Save to CSV
                group.to_csv(reports_dir / f'plus_rate_by_{feature}_{segment_type}.csv')
                print(f"Saved plus rate by {feature} to: {reports_dir / f'plus_rate_by_{feature}_{segment_type}.csv'}")
            else:
                print(f"Column '{feature}' not found in data for this segment. Skipping {feature} analysis.")

if __name__ == "__main__":
    # Configuration
    segment_type_list = ['airport', 'airport_dropoff', 'airport_pickup', 'all', 'churned']
    #segment_type_list = ['all']  # Uncomment to run only for all data
    # Set data version: 'original', 'v2', or 'v3'
    data_version = 'v3'
    main(segment_type_list, data_version) 