import pandas as pd
import matplotlib.pyplot as plt
from load_data import load_parquet_data
from pathlib import Path

def main():
    print("Loading data...")
    df = load_parquet_data()
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Filter for requests where requested_ride_type != preselected_mode
    non_preselected = df[df['requested_ride_type'] != df['preselected_mode']]
    print(f"Found {len(non_preselected)} rows where requested_ride_type != preselected_mode")

    # Count most common requested_ride_type in this subset
    mode_counts = non_preselected['requested_ride_type'].value_counts()
    print("Top 10 most common requested modes (when not preselected):")
    print(mode_counts.head(10))

    # Plot
    plots_dir = Path('/home/sagemaker-user/studio/src/new-rider-v3/plots/eda')
    plots_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(10, 6))
    mode_counts.head(10).plot(kind='bar', color='skyblue')
    plt.title('Most Common Requested Modes (When Not Preselected)')
    plt.xlabel('Requested Ride Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'most_common_non_preselected_modes.pdf')
    print(f"Saved plot to {plots_dir / 'most_common_non_preselected_modes.pdf'}")

if __name__ == "__main__":
    main() 