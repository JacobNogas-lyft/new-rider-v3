# import os
# import sys
# sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src'))
# sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src/offeringsmxmodels'))

# import pandas as pd
# from lyft_distributed import yarn_spark

# from models.main.common.constants import COLUMN_SCORES
# from models.main.common.utils import hash_str
# from models.main.common.utils import MeasureTime

import os
import sys
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src'))
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src/offeringsmxmodels'))

import pandas as pd
from lyft_distributed import yarn_spark

#             return df.toPandas()

if __name__ == "__main__":
    # Read SQL query
    sql_query_path = "/home/sagemaker-user/studio/src/new-rider-v3/queries/simple_test.sql"
    with open(sql_query_path, 'r') as f:
        sql_query = f.read()

    # Define paths and config
    S3_PATH = 's3://data-team/jacobnogas/new-rider-v3/test.parquet'
    
    # Configure Spark session
    spark_config = {
        "cluster": "8*8*8g",
        "spark.driver.maxResultSize": "512g",
        "logging": True,
        "spark.sql.hive.convertMetastoreParquet": "false",
        "spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive": "true",
    }

    # Use yarn_spark directly
    with yarn_spark(spark_config) as session:
        # Execute query
        df = session.sql(sql_query)
        
        # Write to parquet
        df.write.mode("overwrite").parquet(S3_PATH)
        print("Query executed and data saved successfully!")