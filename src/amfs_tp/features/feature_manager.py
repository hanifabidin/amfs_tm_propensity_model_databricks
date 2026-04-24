from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
import logging

class FeatureManager:
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        self.spark = spark
        self.fe = FeatureEngineeringClient()
        self.catalog = catalog
        self.schema = schema

    def register_or_update(self, source_table: str, feature_table_name: str, primary_keys: list, description: str):
        """
        Registers a Silver table as a Feature Table in Unity Catalog.
        """
        full_feature_name = f"{self.catalog}.{self.schema}.{feature_table_name}"
        source_df = self.spark.table(f"{self.catalog}.{self.schema}.{source_table}")

        # Check if feature table already exists
        try:
            self.spark.table(full_feature_name)
            logging.info(f"Updating existing feature table: {full_feature_name}")
            # Mode='merge' handles incremental updates for specific snapshots
            self.fe.write_table(
                name=full_feature_name,
                df=source_df,
                mode="merge"
            )
        except Exception:
            logging.info(f"Creating new feature table: {full_feature_name}")
            self.fe.create_table(
                name=full_feature_name,
                primary_keys=primary_keys,
                df=source_df,
                description=description
            )
        
        return full_feature_name