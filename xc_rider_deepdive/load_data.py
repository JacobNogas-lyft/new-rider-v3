import polars as pl
import s3fs
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
S3_BASE_PATH = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/all_rider_i_new_rider_features_v1_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_b7932c47-ed97-44c2-aebc-cee4c93edfcc'

def get_date_range(s3_base_path):
    """Get the date range from the path name."""
    # Extract dates from the path
    start_date = datetime.strptime(s3_base_path.split('_')[-3], '%Y-%m-%d')
    end_date = datetime.strptime(s3_base_path.split('_')[-2], '%Y-%m-%d')
    return start_date, end_date

def read_parquet_with_polars(pq_file, filesystem):
    """Read a parquet file using Polars for better performance."""
    try:
        # Use s3fs to open the file and read with Polars
        with filesystem.open(pq_file, 'rb') as f:
            df = pl.read_parquet(f)
        return df
    except Exception as e:
        logger.error(f"Error reading parquet file {pq_file}: {str(e)}")
        raise

def load_parquet_data():
    """Load and combine Parquet files from S3 using Polars."""
    s3_base_path = S3_BASE_PATH
    extract_path = f"{s3_base_path}/extract"
    
    logger.info(f"Loading XC rider deepdive data from {s3_base_path}")
    logger.info("Initializing S3 filesystem...")
    fs = s3fs.S3FileSystem()
    
    # Get date range
    start_date, end_date = get_date_range(s3_base_path)
    logger.info(f"Loading data from {start_date.date()} to {end_date.date()}")
    
    # Generate list of expected parquet files
    current_date = start_date
    parquet_files = []
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        parquet_path = f"{extract_path}/{date_str}.parquet"
        
        # Check if file exists
        if fs.exists(parquet_path):
            parquet_files.append(parquet_path)
        else:
            logger.warning(f"No data file found for date: {date_str}")
            
        current_date += timedelta(days=1)
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        raise ValueError("No parquet files found in the specified date range")
    
    # Read and combine using Polars
    logger.info("Reading and combining parquet files with Polars...")
    dfs = []
    
    for pq_file in parquet_files:
        try:
            df = read_parquet_with_polars(pq_file, fs)
            dfs.append(df)
            logger.info(f"Successfully loaded {pq_file}")
        except Exception as e:
            logger.error(f"Error processing {pq_file}: {str(e)}")
            raise
    
    if not dfs:
        raise ValueError("No dataframes were successfully loaded")
    
    # Combine all dataframes using Polars concat with schema alignment
    logger.info("Concatenating dataframes...")
    try:
        # Try normal concat first
        combined_df = pl.concat(dfs, how="vertical")
    except Exception as e:
        logger.warning(f"Normal concat failed: {str(e)}")
        logger.info("Attempting concat with schema alignment...")
        
        # Align schemas by converting problematic columns to consistent types
        aligned_dfs = []
        for df in dfs:
            # Convert large integer columns to float64 for consistency
            df_aligned = df
            for col in df.columns:
                if df[col].dtype == pl.Int64:
                    # Check if this might be a large integer that should be float
                    try:
                        max_val = df[col].max()
                        if max_val and max_val > 9007199254740992:  # JavaScript safe integer limit
                            df_aligned = df_aligned.with_columns(pl.col(col).cast(pl.Float64))
                    except:
                        # If we can't check, convert anyway for safety
                        df_aligned = df_aligned.with_columns(pl.col(col).cast(pl.Float64))
            aligned_dfs.append(df_aligned)
        
        # Try concat again with aligned schemas
        combined_df = pl.concat(aligned_dfs, how="vertical_relaxed")
    
    return combined_df

def get_basic_info(df):
    """Get basic information about the dataset."""
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Number of rows: {df.height}")
    logger.info(f"Number of columns: {df.width}")
    
    # Show column names
    logger.info(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        logger.info(f"  {i+1:3d}. {col}")
    
    # Show data types
    logger.info("\nData types:")
    for col, dtype in zip(df.columns, df.dtypes):
        logger.info(f"  {col}: {dtype}")
    
    return df

def main():
    """Main function to load data and show basic info."""
    try:
        df = load_parquet_data()
        logger.info("✅ Data loading completed successfully!")
        
        # Show basic information
        get_basic_info(df)
        
        # Show first few rows
        logger.info("\nFirst 5 rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    df = main()