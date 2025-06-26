import sys
import os
import importlib.util

# Load the module directly from file path
spec = importlib.util.spec_from_file_location(
    "data_utils", 
    "/home/sagemaker-user/studio/src/offeringsmxmodels/models/main/datagen/library/data_utils.py"
)
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)
execute_query = data_utils.execute_query

# Example SQL query (replace with your actual query as needed)
#query = "SELECT 1 as foo, 'bar' as bar"

query = '''
  SELECT
    ds,
    session_id,
    last_purchase_session_id,
    rider_lyft_id
  FROM core.rider_sessions
  WHERE is_valid_session
    AND last_purchase_session_id IS NOT NULL
    AND ds BETWEEN '2025-04-22' AND '2025-05-22'
'''


if __name__ == "__main__":

    df_trino = execute_query(query, engine_name="trino")
    #df_spark = execute_query(query, engine_name="spark")
    print(df_trino.head())

    # print("--- Trino Example ---")
    # try:
    #     df_trino = execute_query(query, engine_name="trino")
    #     print(df_trino)
    # except Exception as e:
    #     print(f"Trino query failed: {e}")

    # print("\n--- Spark Example ---")
    # try:
    #     df_spark = execute_query(query, engine_name="spark")
    #     print(df_spark)
    # except Exception as e:
    #     print(f"Spark query failed: {e}") 