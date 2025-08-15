import polars as pl
import numpy as np
import logging
from load_data import load_parquet_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to load and explore the XC rider data."""
    logger.info("Starting XC rider data exploration...")
    
    # Load the data
    logger.info("Loading data...")
    df = load_parquet_data()
    logger.info(f"✅ Data loaded successfully: {df.shape}")
    
    # Basic data overview
    logger.info("\n" + "="*60)
    logger.info("BASIC DATA OVERVIEW")
    logger.info("="*60)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Rows: {df.height:,}")
    logger.info(f"Columns: {df.width}")
    
    # Show first few rows
    logger.info("\nFirst 3 rows:")
    print(df.head(3))
    
    # Data types summary
    logger.info(f"\nData types summary:")
    dtype_counts = {}
    for dtype in df.dtypes:
        dtype_str = str(dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
    
    for dtype, count in sorted(dtype_counts.items()):
        logger.info(f"  {dtype}: {count} columns")
    
    return df

if __name__ == "__main__":
    df = main()
