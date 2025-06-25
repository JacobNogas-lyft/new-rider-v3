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
S3_BASE_PATH = 's3://lyft-fugue-cache/expire/90d/datagen/0.0.2/new_rider_features_offeringsmxmodels_relevance_ranking_model_2025-04-22_2025-05-22_f25b3c89-a634-421a-90fe-dc1df09a0098'
EXTRACT_PATH = f"{S3_BASE_PATH}/extract"

def get_date_range():
    """Get the date range from the path name."""
    # Extract dates from the path
    start_date = datetime.strptime(S3_BASE_PATH.split('_')[-3], '%Y-%m-%d')
    end_date = datetime.strptime(S3_BASE_PATH.split('_')[-2], '%Y-%m-%d')
    return start_date, end_date

def standardize_schema(table):
    """Standardize the schema of a table by casting numeric columns to consistent types."""
    schema = table.schema
    new_fields = []
    
    for field in schema:
        if pa.types.is_integer(field.type):
            # Convert all integers to double
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

def load_parquet_data():
    """Load and combine Parquet files from S3."""
    logger.info("Initializing S3 filesystem...")
    fs = s3fs.S3FileSystem()
    
    # Get date range
    start_date, end_date = get_date_range()
    logger.info(f"Loading data from {start_date.date()} to {end_date.date()}")
    
    # Generate list of expected parquet files
    current_date = start_date
    parquet_files = []
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        parquet_path = f"{EXTRACT_PATH}/{date_str}.parquet"
        
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
            table = pq.read_table(pq_file, filesystem=fs)
            # Standardize schema before adding to list
            standardized_table = standardize_schema(table)
            tables.append(standardized_table)
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