
from lyft_data_toolkit.db.read import Query

from lyft_data_toolkit.db.read.df import to_dataframe  # noqa: F401

from lyft_data_toolkit.db.read import Compute
from lyft_data_toolkit.db.read import Query
from lyft_data_toolkit.db.read.df import to_dataframe  # noqa: F401
import os


query_string = """
SELECT user_id,ds,rides_premium_lifetime as rides_premium_lifetime_pre_nov_2023
FROM default.passenger_rides_lifetime
WHERE ds='2023-11-01'
"""


# with open(query_path) as f:
#             query_string = f.read()
df = Query(query_string).execute(compute=Compute.TRINO).to_dataframe()
print(df.head())

# Create the aux_data directory if it doesn't exist
os.makedirs("aux_data", exist_ok=True)

# Save the DataFrame to CSV in the aux_data folder
df.to_csv("aux_data/rides_premium_lifetime_pre_nov_2023.csv", index=False)
