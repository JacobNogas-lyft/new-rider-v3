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
from lyft_distributed import query

#             return df.toPandas()

if __name__ == "__main__":

    S3_PATH = 's3://data-team/jacobnogas/new-rider-v3/test.parquet'
    CLUSTER = '8*8*8g'

    sql_query_path = "/home/sagemaker-user/studio/src/new-rider-v3/queries/simple_test.sql"
    with open(sql_query_path, 'r') as f:
        sql_query = f.read()

    #query_path='/home/sagemaker-user/studio/src/offeringsmxmodels/models/main/relevance/queries/finished_rides_with_scores_v6.sql'
    df=query(sql_query).hive_to_file(S3_PATH, {'cluster': CLUSTER}, to_pandas=True)
    print()