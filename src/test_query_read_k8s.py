import os
import sys
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src'))
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src/offeringsmxmodels'))

import pandas as pd
from lyft_distributed import k8s_spark

if __name__ == "__main__":
    # Read SQL query
    sql_query_path = "/home/sagemaker-user/studio/src/new-rider-v3/queries/simple_test.sql"
    with open(sql_query_path, 'r') as f:
        sql_query = f.read()

    # Define paths and config
    S3_PATH = 's3://data-team/jacobnogas/new-rider-v3/test.parquet'
    
    # Configure Spark session for k8s
    spark_config = {
        "cluster": "8*8*8g",  # This format might need adjustment for k8s
        "spark.driver.maxResultSize": "8g",
        "spark.executor.memory": "8g",
        "spark.executor.cores": "8",
        "spark.driver.memory": "8g",
        "logging": True
    }

    # Use k8s_spark instead of yarn_spark
    with k8s_spark(spark_config) as session:
        # Execute query
        df = session.sql(sql_query)
        
        # Write to parquet
        df.write.mode("overwrite").parquet(S3_PATH)
        print("Query executed and data saved successfully!") 