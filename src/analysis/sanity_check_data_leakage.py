import pandas as pd
import numpy as np
from pathlib import Path
from utils.load_data import load_parquet_data
import warnings
warnings.filterwarnings('ignore')

def add_churned_indicator(df):
    """Add churned user indicator to the dataframe."""
    df['ds'] = pd.to_datetime(df['ds'])
    churned_mask = (
        (df['days_since_signup'] > 365) &
        (df['rides_lifetime'] > 2) &
        (df['all_type_total_rides_365d'] == 0)
    )
    df['is_churned_user'] = churned_mask.astype(int)
    return df

def calculate_plus_percentage(df):
    """Calculate percentage of lifetime rides that were Plus."""
    # Handle division by zero by replacing 0 with NaN, then filling with 0
    df['percent_rides_plus_lifetime'] = (
        df['rides_plus_lifetime'] / df['rides_lifetime'].replace(0, np.nan)
    ).fillna(0)
    
    return df

def create_plus_target(df):
    """Create binary target for Plus requests."""
    # Filter out sessions where Plus was preselected
    df = df[df['preselected_mode'] != 'plus'].copy()
    
    df['chose_plus'] = (df['requested_ride_type'] == 'plus').astype(int)
    return df

def filter_riders_with_multiple_sessions(df, min_sessions=2):
    """
    Filter the dataset to include only riders who have more than the specified 
    number of purchase sessions.
    
    Args:
        df (pandas.DataFrame): Input dataframe with 'rider_lyft_id' and 'purchase_session_id' columns
        min_sessions (int): Minimum number of sessions required (default: 2)
    
    Returns:
        pandas.DataFrame: Filtered dataframe containing only riders with multiple sessions
    """
    # Count sessions per rider
    rider_session_counts = df.groupby('rider_lyft_id')['purchase_session_id'].nunique()
    
    # Get riders with more than min_sessions
    riders_with_multiple_sessions = rider_session_counts[rider_session_counts > min_sessions - 1].index
    
    # Filter the original dataframe
    filtered_df = df[df['rider_lyft_id'].isin(riders_with_multiple_sessions)].copy()
    
    print(f"Original dataset: {len(df):,} rows, {df['rider_lyft_id'].nunique():,} riders")
    print(f"Filtered dataset: {len(filtered_df):,} rows, {filtered_df['rider_lyft_id'].nunique():,} riders")
    print(f"Riders with >{min_sessions-1} sessions: {len(riders_with_multiple_sessions):,}")
    
    return filtered_df

def save_riders_multiple_sessions_data(df, output_dir, min_sessions=2):
    """
    Save a dataframe of riders with more than 1 purchase session, on at least 2 different days,
    and who chose plus at least once, grouped by rider_lyft_id and ds.
    Only include rows where is_finished_ride == True.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Filter for finished rides only
    if 'is_finished_ride' in df.columns:
        df = df[df['is_finished_ride'] == True].copy()
        print(f"Filtered to finished rides only: {len(df)} rows remain.")
    else:
        print("Warning: 'is_finished_ride' column not found in dataframe. No filter applied.")

    # Filter for riders with multiple sessions
    df_filtered = filter_riders_with_multiple_sessions(df, min_sessions)

    # 1. Riders with sessions on at least 2 different days
    days_per_rider = df_filtered.groupby('rider_lyft_id')['ds'].nunique()
    riders_multiple_days = days_per_rider[days_per_rider >= 2].index
    df_filtered = df_filtered[df_filtered['rider_lyft_id'].isin(riders_multiple_days)]

    # 2. Riders who chose plus at least once
    plus_per_rider = df_filtered.groupby('rider_lyft_id')['chose_plus'].sum()
    riders_chose_plus = plus_per_rider[plus_per_rider > 0].index
    df_filtered = df_filtered[df_filtered['rider_lyft_id'].isin(riders_chose_plus)]

    # Select required columns and group by rider_lyft_id and ds
    grouped_df = df_filtered[['ds', 'rider_lyft_id', 'chose_plus', 'rides_plus_lifetime']].groupby(['rider_lyft_id', 'ds']).sum()
    grouped_df = grouped_df.reset_index()
    grouped_df = grouped_df.sort_values(['rider_lyft_id', 'ds'])

    output_file = output_dir / f'riders_multiple_sessions_{min_sessions}plus.csv'
    grouped_df.to_csv(output_file, index=False)

    print(f"Saved riders with {min_sessions}+ sessions (across 2+ days, chose plus at least once, finished rides only) to: {output_file}")
    print(f"Data shape: {grouped_df.shape}")
    print(f"Number of unique riders: {grouped_df['rider_lyft_id'].nunique()}")
    print(f"Date range: {grouped_df['ds'].min()} to {grouped_df['ds'].max()}")

    return grouped_df

def analyze_data_leakage_potential(df_grouped):
    """
    Analyze the potential for data leakage by examining how rider behavior changes over time.
    
    Args:
        df_grouped (pandas.DataFrame): Grouped dataframe with rider_lyft_id, ds, chose_plus, rides_plus_lifetime
    """
    print("\n=== Data Leakage Analysis ===")
    
    # Calculate basic statistics
    total_riders = df_grouped['rider_lyft_id'].nunique()
    total_days = df_grouped['ds'].nunique()
    avg_sessions_per_rider = len(df_grouped) / total_riders
    
    print(f"Total unique riders: {total_riders:,}")
    print(f"Total unique days: {total_days}")
    print(f"Average sessions per rider: {avg_sessions_per_rider:.2f}")
    
    # Analyze temporal patterns
    print(f"\n--- Temporal Analysis ---")
    
    # Check if riders have sessions across multiple days
    rider_day_counts = df_grouped.groupby('rider_lyft_id')['ds'].nunique()
    riders_multiple_days = (rider_day_counts > 1).sum()
    
    print(f"Riders with sessions on multiple days: {riders_multiple_days:,} ({riders_multiple_days/total_riders*100:.1f}%)")
    print(f"Riders with sessions on single day: {total_riders - riders_multiple_days:,} ({(total_riders - riders_multiple_days)/total_riders*100:.1f}%)")
    
    # Analyze Plus choice patterns over time
    print(f"\n--- Plus Choice Patterns ---")
    
    # Calculate Plus rate by day
    daily_plus_rate = df_grouped.groupby('ds')['chose_plus'].mean()
    print(f"Daily Plus rate range: {daily_plus_rate.min():.3f} - {daily_plus_rate.max():.3f}")
    print(f"Daily Plus rate std: {daily_plus_rate.std():.3f}")
    
    # Check for riders with changing Plus behavior
    rider_plus_behavior = df_grouped.groupby('rider_lyft_id')['chose_plus'].agg(['mean', 'std', 'count'])
    riders_with_varying_behavior = (rider_plus_behavior['std'] > 0).sum()
    
    print(f"Riders with varying Plus behavior (std > 0): {riders_with_varying_behavior:,} ({riders_with_varying_behavior/total_riders*100:.1f}%)")
    
    # Analyze rides_plus_lifetime patterns
    print(f"\n--- Plus Lifetime Rides Patterns ---")
    
    # Check if rides_plus_lifetime changes over time for the same rider
    rider_plus_lifetime_stats = df_grouped.groupby('rider_lyft_id')['rides_plus_lifetime'].agg(['mean', 'std', 'min', 'max'])
    riders_with_changing_plus_lifetime = (rider_plus_lifetime_stats['std'] > 0).sum()
    
    print(f"Riders with changing plus_lifetime rides (std > 0): {riders_with_changing_plus_lifetime:,} ({riders_with_changing_plus_lifetime/total_riders*100:.1f}%)")
    
    # Check for potential leakage indicators
    print(f"\n--- Data Leakage Indicators ---")
    
    # Indicator 1: Riders with increasing plus_lifetime over time
    increasing_plus_lifetime = 0
    for rider_id in df_grouped['rider_lyft_id'].unique():
        rider_data = df_grouped[df_grouped['rider_lyft_id'] == rider_id].sort_values('ds')
        if len(rider_data) > 1:
            # Check if plus_lifetime is strictly increasing
            if all(rider_data['rides_plus_lifetime'].iloc[i] <= rider_data['rides_plus_lifetime'].iloc[i+1] 
                   for i in range(len(rider_data)-1)):
                increasing_plus_lifetime += 1
    
    print(f"Riders with strictly increasing plus_lifetime: {increasing_plus_lifetime:,} ({increasing_plus_lifetime/total_riders*100:.1f}%)")
    
    # Indicator 2: Correlation between plus_lifetime and chose_plus within riders
    rider_correlations = []
    for rider_id in df_grouped['rider_lyft_id'].unique():
        rider_data = df_grouped[df_grouped['rider_lyft_id'] == rider_id]
        if len(rider_data) > 1:
            corr = rider_data['rides_plus_lifetime'].corr(rider_data['chose_plus'])
            if not pd.isna(corr):
                rider_correlations.append(corr)
    
    if rider_correlations:
        avg_correlation = np.mean(rider_correlations)
        print(f"Average within-rider correlation (plus_lifetime vs chose_plus): {avg_correlation:.3f}")
        print(f"Riders with positive correlation: {sum(1 for c in rider_correlations if c > 0):,}")
        print(f"Riders with negative correlation: {sum(1 for c in rider_correlations if c < 0):,}")
    
    return {
        'total_riders': total_riders,
        'riders_multiple_days': riders_multiple_days,
        'riders_with_varying_behavior': riders_with_varying_behavior,
        'riders_with_changing_plus_lifetime': riders_with_changing_plus_lifetime,
        'increasing_plus_lifetime': increasing_plus_lifetime,
        'avg_correlation': avg_correlation if rider_correlations else None
    }

def analyze_plus_lifetime_timing(df_grouped):
    """
    Analyze the timing of when rides_plus_lifetime gets updated relative to Plus choices.
    This checks if plus_lifetime increments on the same day as Plus choice or the next day.
    
    Args:
        df_grouped (pandas.DataFrame): Grouped dataframe with rider_lyft_id, ds, chose_plus, rides_plus_lifetime
    """
    print("\n=== Plus Lifetime Timing Analysis ===")
    
    # Convert ds to datetime for proper date arithmetic
    df_grouped['ds'] = pd.to_datetime(df_grouped['ds'])
    
    # Sort by rider_lyft_id and ds
    df_grouped = df_grouped.sort_values(['rider_lyft_id', 'ds'])
    
    # Initialize counters
    same_day_increment = 0
    next_day_increment = 0
    no_increment = 0
    other_patterns = 0
    
    # Track detailed patterns for analysis
    timing_patterns = []
    
    for rider_id in df_grouped['rider_lyft_id'].unique():
        rider_data = df_grouped[df_grouped['rider_lyft_id'] == rider_id].copy()
        
        if len(rider_data) < 2:
            continue
            
        # Sort by date
        rider_data = rider_data.sort_values('ds')
        
        # Check each day for Plus choices and subsequent plus_lifetime changes
        for i in range(len(rider_data) - 1):
            current_day = rider_data.iloc[i]
            next_day = rider_data.iloc[i + 1]
            
            # Check if rider chose Plus on current day
            if current_day['chose_plus'] > 0:
                current_plus_lifetime = current_day['rides_plus_lifetime']
                next_plus_lifetime = next_day['rides_plus_lifetime']
                
                # Calculate the difference
                plus_lifetime_diff = next_plus_lifetime - current_plus_lifetime
                
                # Determine timing pattern
                if plus_lifetime_diff > 0:
                    # Plus lifetime increased
                    if plus_lifetime_diff == current_day['chose_plus']:
                        # Increment matches the number of Plus choices on current day
                        same_day_increment += 1
                        timing_patterns.append({
                            'rider_id': rider_id,
                            'current_date': current_day['ds'],
                            'next_date': next_day['ds'],
                            'chose_plus': current_day['chose_plus'],
                            'current_plus_lifetime': current_plus_lifetime,
                            'next_plus_lifetime': next_plus_lifetime,
                            'pattern': 'same_day_increment'
                        })
                    else:
                        # Increment doesn't match - could be next day or other pattern
                        next_day_increment += 1
                        timing_patterns.append({
                            'rider_id': rider_id,
                            'current_date': current_day['ds'],
                            'next_date': next_day['ds'],
                            'chose_plus': current_day['chose_plus'],
                            'current_plus_lifetime': current_plus_lifetime,
                            'next_plus_lifetime': next_plus_lifetime,
                            'pattern': 'next_day_increment'
                        })
                else:
                    # No increment
                    no_increment += 1
                    timing_patterns.append({
                        'rider_id': rider_id,
                        'current_date': current_day['ds'],
                        'next_date': next_day['ds'],
                        'chose_plus': current_day['chose_plus'],
                        'current_plus_lifetime': current_plus_lifetime,
                        'next_plus_lifetime': next_plus_lifetime,
                        'pattern': 'no_increment'
                    })
    
    # Calculate totals
    total_plus_choices = same_day_increment + next_day_increment + no_increment + other_patterns
    
    print(f"Total Plus choice events analyzed: {total_plus_choices}")
    print(f"Same day increment: {same_day_increment} ({same_day_increment/total_plus_choices*100:.1f}%)")
    print(f"Next day increment: {next_day_increment} ({next_day_increment/total_plus_choices*100:.1f}%)")
    print(f"No increment: {no_increment} ({no_increment/total_plus_choices*100:.1f}%)")
    print(f"Other patterns: {other_patterns} ({other_patterns/total_plus_choices*100:.1f}%)")
    
    # Create detailed analysis dataframe
    timing_df = pd.DataFrame(timing_patterns)
    
    # Additional analysis
    if len(timing_df) > 0:
        print(f"\n--- Detailed Timing Analysis ---")
        
        # Analyze by pattern
        pattern_analysis = timing_df.groupby('pattern').agg({
            'chose_plus': ['count', 'mean', 'sum'],
            'current_plus_lifetime': 'mean',
            'next_plus_lifetime': 'mean'
        }).round(2)
        
        print("Pattern Analysis:")
        print(pattern_analysis)
        
        # Check for exact matches
        exact_matches = timing_df[timing_df['next_plus_lifetime'] - timing_df['current_plus_lifetime'] == timing_df['chose_plus']]
        print(f"\nExact matches (increment = chose_plus): {len(exact_matches)} ({len(exact_matches)/len(timing_df)*100:.1f}%)")
        
        # Check for partial matches
        partial_matches = timing_df[
            (timing_df['next_plus_lifetime'] - timing_df['current_plus_lifetime'] > 0) & 
            (timing_df['next_plus_lifetime'] - timing_df['current_plus_lifetime'] != timing_df['chose_plus'])
        ]
        print(f"Partial matches (increment > 0 but != chose_plus): {len(partial_matches)} ({len(partial_matches)/len(timing_df)*100:.1f}%)")
    
    return {
        'same_day_increment': same_day_increment,
        'next_day_increment': next_day_increment,
        'no_increment': no_increment,
        'other_patterns': other_patterns,
        'total_plus_choices': total_plus_choices,
        'timing_df': timing_df if len(timing_patterns) > 0 else pd.DataFrame()
    }

def analyze_plus_lifetime_timing_refined(df_grouped):
    """
    Refined timing analysis: Only consider days where a Plus ride was taken (chose_plus > 0),
    and compare rides_plus_lifetime to the previous day's value for the same rider.
    """
    print("\n=== Refined Plus Lifetime Timing Analysis (Only Plus Days) ===")
    df_grouped = df_grouped.sort_values(['rider_lyft_id', 'ds'])
    df_grouped['ds'] = pd.to_datetime(df_grouped['ds'])

    # Only consider days where a Plus ride was taken
    plus_days = df_grouped[df_grouped['chose_plus'] > 0].copy()

    # For each such day, get the previous day's rides_plus_lifetime
    plus_days['prev_plus_lifetime'] = (
        plus_days.groupby('rider_lyft_id')['rides_plus_lifetime'].shift(1)
    )

    # Calculate the increment
    plus_days['lifetime_increment'] = plus_days['rides_plus_lifetime'] - plus_days['prev_plus_lifetime']

    # Now analyze the increments
    same_day = (plus_days['lifetime_increment'] == plus_days['chose_plus']).sum()
    partial = ((plus_days['lifetime_increment'] > 0) & (plus_days['lifetime_increment'] != plus_days['chose_plus'])).sum()
    no_increment = (plus_days['lifetime_increment'] <= 0).sum()

    print(f"Total Plus choice days analyzed: {len(plus_days)}")
    print(f"Same day increment: {same_day} ({same_day/len(plus_days)*100:.1f}%)")
    print(f"Partial increment: {partial} ({partial/len(plus_days)*100:.1f}%)")
    print(f"No increment: {no_increment} ({no_increment/len(plus_days)*100:.1f}%)")

    # Optionally, return the DataFrame for further inspection
    return plus_days

def main(data_version='v2', min_sessions=2):
    """
    Main function to perform data leakage sanity check.
    
    Args:
        data_version (str): Data version to use ('original', 'v2', or 'v3') (default: 'v2')
        min_sessions (int): Minimum number of sessions required (default: 2)
    """
    print("Starting Data Leakage Sanity Check")
    print(f"Using {data_version.upper()} data")
    print(f"Minimum sessions per rider: {min_sessions}")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = load_parquet_data(data_version)
    df = add_churned_indicator(df)
    
    # Basic filtering
    required_cols = ['requested_ride_type', 'rides_lifetime', 'rides_plus_lifetime', 'preselected_mode', 
                     'plus_availability_caveat', 'standard_saver_availability_caveat']
    df = df.dropna(subset=required_cols)
    
    # Calculate Plus percentage and create target
    df = calculate_plus_percentage(df)
    df = create_plus_target(df)
    
    print(f"Data loaded: {len(df)} rows, {df['rider_lyft_id'].nunique()} riders")
    
    # Save riders with multiple sessions data
    print("\nSaving riders with multiple sessions data...")
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
    data_dir = base_path / 'data/riders_multiple_sessions'
    grouped_df = save_riders_multiple_sessions_data(df, data_dir, min_sessions)
    
    # Analyze potential data leakage
    leakage_analysis = analyze_data_leakage_potential(grouped_df)
    
    # Analyze Plus lifetime timing
    timing_analysis = analyze_plus_lifetime_timing(grouped_df)
    
    # Analyze refined Plus lifetime timing
    refined_timing_analysis = analyze_plus_lifetime_timing_refined(grouped_df)
    
    # Save analysis summary
    summary_file = data_dir / f'leakage_analysis_summary_{min_sessions}plus.txt'
    with open(summary_file, 'w') as f:
        f.write("Data Leakage Analysis Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Data version: {data_version.upper()}\n")
        f.write(f"Minimum sessions per rider: {min_sessions}\n\n")
        
        f.write("Key Metrics:\n")
        f.write(f"- Total unique riders: {leakage_analysis['total_riders']:,}\n")
        f.write(f"- Riders with sessions on multiple days: {leakage_analysis['riders_multiple_days']:,}\n")
        f.write(f"- Riders with varying Plus behavior: {leakage_analysis['riders_with_varying_behavior']:,}\n")
        f.write(f"- Riders with changing plus_lifetime: {leakage_analysis['riders_with_changing_plus_lifetime']:,}\n")
        f.write(f"- Riders with increasing plus_lifetime: {leakage_analysis['increasing_plus_lifetime']:,}\n")
        
        if leakage_analysis['avg_correlation'] is not None:
            f.write(f"- Average within-rider correlation: {leakage_analysis['avg_correlation']:.3f}\n")
        
        f.write(f"\nPlus Lifetime Timing Analysis:\n")
        f.write(f"- Total Plus choice events: {timing_analysis['total_plus_choices']:,}\n")
        f.write(f"- Same day increment: {timing_analysis['same_day_increment']:,} ({timing_analysis['same_day_increment']/timing_analysis['total_plus_choices']*100:.1f}%)\n")
        f.write(f"- Next day increment: {timing_analysis['next_day_increment']:,} ({timing_analysis['next_day_increment']/timing_analysis['total_plus_choices']*100:.1f}%)\n")
        f.write(f"- No increment: {timing_analysis['no_increment']:,} ({timing_analysis['no_increment']/timing_analysis['total_plus_choices']*100:.1f}%)\n")
    
    # Save detailed timing analysis
    if len(timing_analysis['timing_df']) > 0:
        timing_file = data_dir / f'plus_lifetime_timing_analysis_{min_sessions}plus.csv'
        timing_analysis['timing_df'].to_csv(timing_file, index=False)
        print(f"Detailed timing analysis saved to: {timing_file}")
    
    print(f"\nAnalysis summary saved to: {summary_file}")
    print("\nData leakage sanity check complete!")

if __name__ == "__main__":
    # Configuration
    data_version = 'v3'  # Set to 'original', 'v2', or 'v3'
    min_sessions = 2  # Minimum number of sessions per rider
    
    main(data_version, min_sessions) 