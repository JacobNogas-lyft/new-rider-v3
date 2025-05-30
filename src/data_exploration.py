import os
import sys

# Add both the src directory and offeringsmxmodels directory to the Python path
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src'))
sys.path.insert(0, os.path.abspath('/home/sagemaker-user/studio/src/offeringsmxmodels'))

from offeringsmxmodels.models.main.relevance.relevance_model_v6 import Model 
from models.main.relevance.library.lls_hyperparameters import RELEVANCE_MODEL_LLS_HYPERPARAMETERS
print('hi')


hp = {
    row['name']: row['default_value']
    for row in RELEVANCE_MODEL_LLS_HYPERPARAMETERS
} 
hp['s3_dir'] += 'vduzhik/test_v6'
hp['in_notebook'] = True
hp['total_rows_limit'] = 1_000_000

hp.update({
    'weight_plus': 0.7,
    'weight_premium': 2.1,
    'weight_lux': 8.0,
    'weight_luxsuv': 15.0,
})

model = Model(hp)
print()
#clf = model.train()

#/home/sagemaker-user/studio/src/offeringsmxmodels/models/main/relevance/queries/finished_rides_with_scores_v6.sql