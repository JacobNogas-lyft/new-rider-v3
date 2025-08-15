import pyarrow.parquet as pq
import pyarrow as pa
import s3fs
import pandas as pd
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
S3_BASE_PATH_ORIGINAL = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/new_rider_features_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_f25b3c89-a634-421a-90fe-dc1df09a0098'
S3_BASE_PATH_V2 = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/new_rider_features_v2_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_5dbd8727-c741-4645-a236-a7d702587ea1'
#S3_BASE_PATH_V3 = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/new_rider_features_v3_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_728c89c7-1fc8-46c2-aa4b-f9f08ce9a384'
#S3_BASE_PATH_V3 = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/new_rider_features_v3_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_19f1b346-bb03-4761-8eee-b9d09a88e843'
S3_BASE_PATH_V3 = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/new_rider_features_v3_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_61213c65-d78d-4fd5-840d-dc7da991b098'
S3_BASE_PATH_V4 = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/new_rider_features_v4_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_95a9c5c7-bb41-4870-b646-1d0f48b50efa'

EXTRACT_PATH = f"{S3_BASE_PATH_ORIGINAL}/extract"

def get_date_range(s3_base_path):
    """Get the date range from the path name."""
    # Extract dates from the path
    start_date = datetime.strptime(s3_base_path.split('_')[-3], '%Y-%m-%d')
    end_date = datetime.strptime(s3_base_path.split('_')[-2], '%Y-%m-%d')
    return start_date, end_date

def standardize_schema(table):
    """Standardize the schema of a table by casting numeric columns to consistent types."""
    schema = table.schema
    new_fields = []
    
    for field in schema:
        if pa.types.is_integer(field.type):
            # Convert all integers to double to avoid overflow issues
            new_fields.append(pa.field(field.name, pa.float64()))
        elif pa.types.is_floating(field.type):
            # Keep doubles as is
            new_fields.append(field)
        elif pa.types.is_null(field.type):
            # Convert null to double
            new_fields.append(pa.field(field.name, pa.float64()))
        else:
            # Keep non-numeric types as is
            new_fields.append(field)
    
    new_schema = pa.schema(new_fields)
    return table.cast(new_schema)

def safe_read_parquet(pq_file, filesystem):
    """Safely read a parquet file, handling large integers."""
    try:
        # First, try to read the table with schema inference disabled for integers
        table = pq.read_table(pq_file, filesystem=filesystem)
        
        # Check for problematic large integers and convert them
        schema = table.schema
        new_fields = []
        
        for i, field in enumerate(schema):
            if pa.types.is_integer(field.type):
                # Get the column data
                column = table.column(i)
                
                # Check if any values are outside safe range
                if field.type in [pa.int64(), pa.uint64()]:
                    # Convert large integers to float64 to avoid overflow
                    new_fields.append(pa.field(field.name, pa.float64()))
                else:
                    # Smaller integers can be converted to float64
                    new_fields.append(pa.field(field.name, pa.float64()))
            elif pa.types.is_floating(field.type):
                new_fields.append(field)
            elif pa.types.is_null(field.type):
                new_fields.append(pa.field(field.name, pa.float64()))
            else:
                new_fields.append(field)
        
        new_schema = pa.schema(new_fields)
        return table.cast(new_schema)
        
    except Exception as e:
        logger.error(f"Error in safe_read_parquet for {pq_file}: {str(e)}")
        raise

def get_s3_base_path(data_version='original'):
    """Get the appropriate S3 base path based on data version.
    
    Args:
        data_version (str): Data version to load. Options: 'original', 'v2', 'v3', 'v4'
    """
    if data_version == 'v2':
        return S3_BASE_PATH_V2
    elif data_version == 'v3':
        return S3_BASE_PATH_V3
    elif data_version == 'v4':
        return S3_BASE_PATH_V4
    else:
        return S3_BASE_PATH_ORIGINAL

def load_parquet_data(data_version='original'):
    """Load and combine Parquet files from S3.
    
    Args:
        data_version (str): Data version to load. Options: 'original', 'v2', 'v3', 'v4'
    """
    # Choose the appropriate S3 base path
    s3_base_path = get_s3_base_path(data_version)
    extract_path = f"{s3_base_path}/extract"
    
    logger.info(f"Loading {data_version.upper()} data from {s3_base_path}")
    
    try:
        # Try the pandas direct approach first (better for handling large integers)
        logger.info("Attempting to load data using pandas direct approach...")
        return load_parquet_data_pandas_direct(extract_path, s3_base_path)
    except Exception as e:
        logger.warning(f"Pandas direct approach failed: {str(e)}")
        logger.info("Falling back to PyArrow approach...")
        
        # Fall back to PyArrow approach
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
        
        # Read and combine
        logger.info("Reading and combining parquet files...")
        tables = []
        for pq_file in parquet_files:
            try:
                table = safe_read_parquet(pq_file, fs)
                tables.append(table)
                logger.info(f"Successfully loaded and standardized {pq_file}")
            except Exception as e:
                logger.error(f"Error processing {pq_file}: {str(e)}")
                raise
        
        if not tables:
            raise ValueError("No tables were successfully loaded")
        
        combined_table = pa.concat_tables(tables)
        
        # Convert to pandas DataFrame for easier analysis
        df = combined_table.to_pandas()
        
        return df

def load_parquet_data_pandas_direct(extract_path, s3_base_path):
    """Alternative loading method using pandas directly with dtype specification."""
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
    
    # Read and combine using pandas directly
    logger.info("Reading and combining parquet files with pandas...")
    dfs = []
    
    for pq_file in parquet_files:
        try:
            # Use pandas to read with string_to_object=False to handle large integers
            with fs.open(pq_file, 'rb') as f:
                df = pd.read_parquet(f, engine='pyarrow')
                
                # Convert any large integer columns to float64
                for col in df.columns:
                    if pd.api.types.is_integer_dtype(df[col]):
                        # Check if any values are outside safe range for JavaScript
                        try:
                            col_max = df[col].max()
                            col_min = df[col].min()
                            if col_max > 9007199254740992 or col_min < -9007199254740992:
                                logger.info(f"Converting column {col} from {df[col].dtype} to float64 due to large values")
                                df[col] = df[col].astype('float64')
                        except Exception:
                            # If we can't check the range, convert to float64 anyway for safety
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
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

def main():
    """Main function to orchestrate the data loading process."""
    try:
        df = load_parquet_data()
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        
        # Basic data analysis
        logger.info("\nData Summary:")
        logger.info(f"Number of rows: {len(df)}")
        logger.info(f"Number of columns: {len(df.columns)}")
        logger.info("\nColumn names:")
        for col in df.columns:
            logger.info(f"- {col}")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 