from amfs_tp.data_sources.bm_handler import BMHandler
from amfs_tp.data_sources.amfs_handler import AMFSHandler

class InferenceJob:
    def __init__(self, spark, config, snapshot):
        self.spark = spark
        self.bm_handler = BMHandler(spark, config)
        self.amfs_handler = AMFSHandler(spark, config)
        self.snapshot = snapshot

    def run(self):
        # 1. Fetch Cleaned & Validated Data
        # If BM or AMFS data is bad, the Handlers will raise an error here
        df_bm = self.bm_handler.get_data(self.snapshot)
        df_amfs = self.amfs_handler.get_data(self.snapshot)

        # 2. Proceed to Feature Joining & Scoring
        # ... logic continues