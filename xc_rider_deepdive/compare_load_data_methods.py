import polars as pl
import pandas as pd
import s3fs
import logging
import time
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

def load_parquet_data_with_polars(extract_path, s3_base_path):
    """Load data using Polars for better performance."""
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

def load_parquet_data_with_pandas(extract_path, s3_base_path):
    """Load data using pandas for comparison."""
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
    
    # Read and combine using pandas
    logger.info("Reading and combining parquet files with pandas...")
    dfs = []
    
    for pq_file in parquet_files:
        try:
            with fs.open(pq_file, 'rb') as f:
                df = pd.read_parquet(f, engine='pyarrow')
                
                # Convert any large integer columns to float64
                for col in df.columns:
                    if pd.api.types.is_integer_dtype(df[col]):
                        try:
                            col_max = df[col].max()
                            col_min = df[col].min()
                            if col_max > 9007199254740992 or col_min < -9007199254740992:
                                logger.info(f"Converting column {col} from {df[col].dtype} to float64 due to large values")
                                df[col] = df[col].astype('float64')
                        except Exception:
                            logger.info(f"Converting column {col} from {df[col].dtype} to float64 for safety")
                            df[col] = df[col].astype('float64')
                
                dfs.append(df)
                logger.info(f"Successfully loaded {pq_file}")
        except Exception as e:
            logger.error(f"Error processing {pq_file}: {str(e)}")
            raise
    
    if not dfs:
        raise ValueError("No dataframes were successfully loaded")
    
    # Combine all dataframes
    logger.info("Concatenating dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

def load_parquet_data(use_polars=True):
    """Load and combine Parquet files from S3.
    
    Args:
        use_polars (bool): If True, use Polars for loading (faster). If False, use pandas.
    
    Returns:
        DataFrame: Polars DataFrame if use_polars=True, pandas DataFrame if use_polars=False
    """
    s3_base_path = S3_BASE_PATH
    extract_path = f"{s3_base_path}/extract"
    
    logger.info(f"Loading XC rider deepdive data from {s3_base_path}")
    
    try:
        if use_polars:
            logger.info("Loading data using Polars...")
            return load_parquet_data_with_polars(extract_path, s3_base_path)
        else:
            logger.info("Loading data using pandas...")
            return load_parquet_data_with_pandas(extract_path, s3_base_path)
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def compare_loading_performance():
    """Compare loading performance between Polars and pandas."""
    logger.info("=" * 60)
    logger.info("PERFORMANCE COMPARISON: Polars vs Pandas")
    logger.info("=" * 60)
    
    # Test Polars loading
    logger.info("\n🚀 Testing Polars loading...")
    start_time = time.time()
    try:
        df_polars = load_parquet_data(use_polars=True)
        polars_time = time.time() - start_time
        logger.info(f"✅ Polars loading completed in {polars_time:.2f} seconds")
        logger.info(f"   Shape: {df_polars.shape}")
    except Exception as e:
        logger.error(f"❌ Polars loading failed: {str(e)}")
        df_polars = None
        polars_time = None
    
    # Test pandas loading
    logger.info("\n🐼 Testing pandas loading...")
    start_time = time.time()
    try:
        df_pandas = load_parquet_data(use_polars=False)
        pandas_time = time.time() - start_time
        logger.info(f"✅ Pandas loading completed in {pandas_time:.2f} seconds")
        logger.info(f"   Shape: {df_pandas.shape}")
    except Exception as e:
        logger.error(f"❌ Pandas loading failed: {str(e)}")
        df_pandas = None
        pandas_time = None
    
    # Compare results
    if polars_time and pandas_time:
        speedup = pandas_time / polars_time
        logger.info(f"\n📊 PERFORMANCE SUMMARY:")
        logger.info(f"   Polars: {polars_time:.2f} seconds")
        logger.info(f"   Pandas: {pandas_time:.2f} seconds")
        logger.info(f"   Speedup: {speedup:.2f}x faster with Polars")
        
        if speedup > 1:
            logger.info(f"   🏆 Polars is {speedup:.2f}x faster!")
        else:
            logger.info(f"   🐼 Pandas is {1/speedup:.2f}x faster!")
    
    return df_polars if df_polars is not None else df_pandas

def main(compare_performance=False):
    """Main function to orchestrate the data loading process."""
    try:
        if compare_performance:
            df = compare_loading_performance()
        else:
            # Default to Polars loading
            df = load_parquet_data(use_polars=True)
        
        if df is not None:
            # Determine if it's Polars or pandas DataFrame
            is_polars = hasattr(df, 'height')
            
            if is_polars:
                logger.info(f"Successfully loaded Polars DataFrame with shape: {df.shape}")
                logger.info(f"Number of rows: {df.height}")
                logger.info(f"Number of columns: {df.width}")
            else:
                logger.info(f"Successfully loaded pandas DataFrame with shape: {df.shape}")
                logger.info(f"Number of rows: {len(df)}")
                logger.info(f"Number of columns: {len(df.columns)}")
            
            # Basic data analysis
            logger.info("\nColumn names:")
            for col in df.columns:
                logger.info(f"- {col}")
            
            # Show data types
            logger.info("\nData types:")
            logger.info(df.dtypes)
            
            # Show first few rows
            logger.info("\nFirst 5 rows:")
            logger.info(df.head())
            
            # Show basic statistics (if available)
            try:
                logger.info("\nBasic statistics:")
                logger.info(df.describe())
            except Exception as e:
                logger.info(f"Could not generate statistics: {str(e)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    df = main(compare_performance=True)
