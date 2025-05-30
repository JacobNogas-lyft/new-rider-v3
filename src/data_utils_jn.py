import os
import sys
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src'))
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src/offeringsmxmodels'))

import pandas as pd
from lyft_distributed import yarn_spark

from models.main.common.constants import COLUMN_SCORES
import os
from models.main.common.utils import hash_str
from models.main.common.utils import MeasureTime

from lyft_distributed import query

# def get_query_template(abs_filepath: str) -> str:
#     """Reads the query file at filepath."""
#     with open(abs_filepath) as file:
#         query = file.read()
#     return query


# def get_query_str(params: dict) -> str:
#     """Reads the query file at filepath."""
#     template = get_query_template(params['query_path'])
#     return template.format(**params['query_params'])


# def query_data(
#     params: dict,
#     log,
# ) -> pd.DataFrame:
#     """
#     Runs the query on a cluster via Spark with high-resources demanding default config.

#     Returns:
#         pandas.DataFrame of the dataset
#     """

#     query_string = get_query_str(params)
#     query_hash = hash_str(query_string) + ".parquet"
#     data_path = os.path.join(params['s3_dir'], query_hash)

#     default_config = {
#         "cluster": "256*16*8g",
#         "spark.driver.maxResultSize": "512g",
#         "logging": True,
#         "spark.sql.hive.convertMetastoreParquet": "false",
#         "spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive": "true",
#     }

#     with yarn_spark(dict(default_config, **params['lyft_distributed_config'])) as session:
#         # try:
#         #     log(f"Trying to load data from s3: {data_path}")
#         #     with MeasureTime("ReadParquet", True):
#         #         return pd.read_parquet(data_path, engine="pyarrow", use_threads=True)
#         # except Exception:
#         #     log("Existing data not found.")

#         with MeasureTime("SQL query", True):
#             df = session.sql(query_string).repartition("ds")
#             df.write.partitionBy("ds").format("parquet").mode(
#                 "overwrite",
#             ).save(
#                 data_path,
#             )
#             return df.toPandas()

if __name__ == "__main__":
    params = {
        "query_path": "/home/sagemaker-user/studio/src/new-rider-v3/queries/simple_test.sql",
        "query_params": {},
        "s3_dir": "s3://lyft-fugue-cache/expire/30d/relevance_model/jnogas/new_rider_v3/2025-05-30",
        "lyft_distributed_config": {"cluster": "256*16*8g"}
    }
    # df= query_data(params, print)

    # print(df.head())


    sql_query = params['query_path']


    S3_PATH = 's3://data-team/jacobnogas/new-rider-v3/test.parquet'
    CLUSTER = '64*16*8g'
    # query(sql_query).hive_to_file(S3_PATH, {'cluster': CLUSTER})
    # print('done')


    query_path='/home/sagemaker-user/studio/src/offeringsmxmodels/models/main/relevance/queries/finished_rides_with_scores_v6.sql'
    query(sql_query).hive_to_file(S3_PATH, {'cluster': CLUSTER})