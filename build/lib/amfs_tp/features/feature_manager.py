from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession
import logging

class FeatureManager:
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        self.spark = spark
        self.fe = FeatureEngineeringClient()
        self.catalog = catalog
        self.schema = schema         # This is the 'clean' schema (Input)
        self.target_schema = None    # Will be set to 'features' by the FeatureJob

    def register_or_update(self, source_table: str, feature_table_name: str, primary_keys: list, description: str):
        """
        Registers or updates a Feature Table in Unity Catalog.
        
        Logic:
        - Reads from: {catalog}.{clean_schema}
        - Writes to: {catalog}.{features_schema} (via target_schema injection)
        """
        # 1. Determine Target Destination (Priority: target_schema -> schema)
        schema_out = self.target_schema if self.target_schema else self.schema
        full_feature_name = f"{self.catalog}.{schema_out}.{feature_table_name}"
        
        # 2. Resolve Source DataFrame
        # Your feature classes often create TempViews (e.g., 'temp_trx_results') 
        # before calling this method.
        if "." not in source_table:
            # Case A: Source is a TempView or local table name
            logging.info(f"Loading source from local view/table: {source_table}")
            source_df = self.spark.table(source_table)
        else:
            # Case B: Source is a fully qualified table string (e.g., 'amfs_tm.clean.table')
            # If the user passed a dot, we assume they know the full path
            logging.info(f"Loading source from qualified path: {source_table}")
            source_df = self.spark.table(source_table)

        # 3. Feature Engineering Client Operations
        try:
            # Check if the feature table already exists in the 'features' schema
            self.spark.table(full_feature_name)
            
            logging.info(f"Feature table exists. Merging data into: {full_feature_name}")
            # write_table with mode='merge' is best for propensity models 
            # as it updates existing CIFs for the current snapshot.
            self.fe.write_table(
                name=full_feature_name,
                df=source_df,
                mode="merge"
            )
            
        except Exception as e:
            # If table doesn't exist (or another error occurs), create it for the first time
            if "TABLE_OR_VIEW_NOT_FOUND" in str(e) or "cannot be found" in str(e).lower():
                logging.info(f"Feature table not found. Creating new table: {full_feature_name}")
                self.fe.create_table(
                    name=full_feature_name,
                    primary_keys=primary_keys,
                    df=source_df,
                    description=description
                )
            else:
                # Re-raise if it's a genuine connection/permission error
                logging.error(f"Error during feature table registration: {str(e)}")
                raise e
        
        return full_feature_name