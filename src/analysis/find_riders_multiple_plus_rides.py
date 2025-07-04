import pandas as pd
import numpy as np
from pathlib import Path
from utils.load_data import load_parquet_data
import warnings
warnings.filterwarnings('ignore')

def find_riders_multiple_plus_rides(df, data_version='v3'):
    """
    Find riders who have multiple rides with requested_ride_type = 'plus' and save their data.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        data_version (str): Data version being used for output naming
    
    Returns:
        pandas.DataFrame: Filtered dataframe with only riders who have multiple Plus rides
    """
    print("Finding riders with multiple Plus rides...")
    
    # Filter for Plus rides only
    plus_rides = df[df['requested_ride_type'] == 'plus'].copy()
    print(f"Total Plus rides: {len(plus_rides)}")
    
    # Count Plus rides per rider
    plus_rides_per_rider = plus_rides.groupby('rider_lyft_id').size()
    riders_multiple_plus = plus_rides_per_rider[plus_rides_per_rider > 1].index
    
    print(f"Riders with multiple Plus rides: {len(riders_multiple_plus)}")
    print(f"Plus rides per rider (multiple riders):")
    print(plus_rides_per_rider[plus_rides_per_rider > 1].describe())
    
    # Get all data for riders with multiple Plus rides
    riders_multiple_plus_data = df[df['rider_lyft_id'].isin(riders_multiple_plus)].copy()
    
    # Create chose_plus column
    riders_multiple_plus_data['chose_plus'] = (riders_multiple_plus_data['requested_ride_type'] == 'plus').astype(int)
    
    # Select only the required columns
    required_columns = ['rider_lyft_id', 'ds', 'purchase_session_id', 'chose_plus', 'rides_plus_lifetime']
    riders_multiple_plus_data = riders_multiple_plus_data[required_columns].copy()
    
    # Sort by rider_lyft_id, ds, and purchase_session_id
    riders_multiple_plus_data = riders_multiple_plus_data.sort_values(
        ['rider_lyft_id', 'ds', 'purchase_session_id']
    )
    
    print(f"Total rows for riders with multiple Plus rides: {len(riders_multiple_plus_data)}")
    print(f"Unique riders in output: {riders_multiple_plus_data['rider_lyft_id'].nunique()}")
    
    return riders_multiple_plus_data

def save_riders_multiple_plus_data(df, output_dir, data_version='v3'):
    """
    Save the dataframe of riders with multiple Plus rides to CSV.
    
    Args:
        df (pandas.DataFrame): Dataframe to save
        output_dir (str or Path): Output directory
        data_version (str): Data version for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert rider_lyft_id to string to avoid scientific notation
    df = df.copy()
    df['rider_lyft_id'] = df['rider_lyft_id'].astype(str)
    
    # Create filename with data version
    output_file = output_dir / f'riders_multiple_plus_rides_{data_version}.csv'
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Saved data to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024**2:.2f} MB")
    
    return output_file

def analyze_plus_ride_patterns(df):
    """
    Analyze patterns in Plus ride behavior for riders with multiple Plus rides.
    
    Args:
        df (pandas.DataFrame): Dataframe with riders who have multiple Plus rides
    """
    print("\n=== Plus Ride Pattern Analysis ===")
    
    # Basic statistics
    total_riders = df['rider_lyft_id'].nunique()
    total_rides = len(df)
    plus_rides = len(df[df['chose_plus'] == 1])
    
    print(f"Total riders: {total_riders:,}")
    print(f"Total rides: {total_rides:,}")
    print(f"Plus rides: {plus_rides:,} ({plus_rides/total_rides*100:.1f}%)")
    
    # Plus rides per rider
    plus_rides_per_rider = df[df['chose_plus'] == 1].groupby('rider_lyft_id').size()
    print(f"\nPlus rides per rider:")
    print(f"  Mean: {plus_rides_per_rider.mean():.2f}")
    print(f"  Median: {plus_rides_per_rider.median():.1f}")
    print(f"  Min: {plus_rides_per_rider.min()}")
    print(f"  Max: {plus_rides_per_rider.max()}")
    
    # Date range analysis
    df['ds'] = pd.to_datetime(df['ds'])
    date_range = df['ds'].max() - df['ds'].min()
    print(f"\nDate range: {df['ds'].min()} to {df['ds'].max()} ({date_range.days} days)")
    
    # Plus rides over time
    daily_plus_rides = df[df['chose_plus'] == 1].groupby('ds').size()
    print(f"\nDaily Plus rides:")
    print(f"  Mean per day: {daily_plus_rides.mean():.2f}")
    print(f"  Max in a day: {daily_plus_rides.max()}")
    
    return {
        'total_riders': total_riders,
        'total_rides': total_rides,
        'plus_rides': plus_rides,
        'plus_rides_per_rider': plus_rides_per_rider,
        'daily_plus_rides': daily_plus_rides
    }

def main(data_version='v3'):
    """
    Main function to find riders with multiple Plus rides and save their data.
    
    Args:
        data_version (str): Data version to use ('original', 'v2', or 'v3')
    """
    print("Finding Riders with Multiple Plus Rides")
    print(f"Using {data_version.upper()} data")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = load_parquet_data(data_version)
    print(f"Loaded {len(df):,} rows, {df['rider_lyft_id'].nunique():,} riders")

    # Ensure rider_lyft_id is string with full precision
    if df['rider_lyft_id'].dtype == float:
        if df['rider_lyft_id'].isnull().sum() == 0:
            df['rider_lyft_id'] = df['rider_lyft_id'].astype('int64').astype(str)
        else:
            df = df.dropna(subset=['rider_lyft_id'])
            df['rider_lyft_id'] = df['rider_lyft_id'].astype('int64').astype(str)
    else:
        df['rider_lyft_id'] = df['rider_lyft_id'].astype(str)
    
    # Basic filtering
    required_cols = ['rider_lyft_id', 'requested_ride_type', 'ds', 'purchase_session_id', 'rides_plus_lifetime']
    df = df.dropna(subset=required_cols)
    print(f"After filtering for required columns: {len(df):,} rows")
    
    # Find riders with multiple Plus rides
    riders_multiple_plus_data = find_riders_multiple_plus_rides(df, data_version)
    
    # Save the data
    print("\nSaving data...")
    base_path = Path('/home/sagemaker-user/studio/src/new-rider-v3')
    output_dir = base_path / 'data/riders_multiple_plus_rides'
    output_file = save_riders_multiple_plus_data(riders_multiple_plus_data, output_dir, data_version)
    
    # Analyze patterns
    analysis_results = analyze_plus_ride_patterns(riders_multiple_plus_data)
    
    # Save analysis summary
    summary_file = output_dir / f'plus_rides_analysis_summary_{data_version}.txt'
    with open(summary_file, 'w') as f:
        f.write("Riders with Multiple Plus Rides - Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data version: {data_version.upper()}\n\n")
        
        f.write("Key Metrics:\n")
        f.write(f"- Total riders: {analysis_results['total_riders']:,}\n")
        f.write(f"- Total rides: {analysis_results['total_rides']:,}\n")
        f.write(f"- Plus rides: {analysis_results['plus_rides']:,}\n")
        f.write(f"- Plus ride percentage: {analysis_results['plus_rides']/analysis_results['total_rides']*100:.1f}%\n\n")
        
        f.write("Plus Rides per Rider:\n")
        f.write(f"- Mean: {analysis_results['plus_rides_per_rider'].mean():.2f}\n")
        f.write(f"- Median: {analysis_results['plus_rides_per_rider'].median():.1f}\n")
        f.write(f"- Min: {analysis_results['plus_rides_per_rider'].min()}\n")
        f.write(f"- Max: {analysis_results['plus_rides_per_rider'].max()}\n\n")
        

    
    print(f"\nAnalysis summary saved to: {summary_file}")
    print(f"Data saved to: {output_file}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Configuration
    data_version = 'v3'  # Set to 'original', 'v2', or 'v3'
    
    main(data_version) 