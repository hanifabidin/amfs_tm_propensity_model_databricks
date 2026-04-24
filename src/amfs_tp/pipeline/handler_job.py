import logging
from amfs_tp.data_sources.bm_handler import BMHandler
from amfs_tp.data_sources.amfs_handler import AMFSHandler

class HandlerJob:
    def __init__(self, spark, catalog="amfs_tm"):
        self.spark = spark
        self.catalog = catalog
        self.raw_schema = f"{catalog}.raw"
        self.clean_schema = f"{catalog}.clean"
        self.logger = logging.getLogger(__name__)
        
        # Initialize Handlers
        self.bm_handler = BMHandler(spark, catalog=catalog)
        self.amfs_handler = AMFSHandler(spark, catalog=catalog)

    def _get_tables_by_pattern(self, prefix):
        """
        Dynamically finds tables in the raw schema matching prefix and _raw suffix.
        Returns a list of base names (e.g., 'bm_phsumm_raw' -> 'phsumm')
        """
        tables = self.spark.catalog.listTables(self.raw_schema)
        # Filter for prefix (bm_ or amfs_) and suffix (_raw)
        matched_tables = [
            t.name for t in tables 
            if t.name.startswith(f"{prefix}_") and t.name.endswith("_raw")
        ]
        
        # Extract the middle part: 'bm_phsumm_raw' -> 'phsumm'
        base_names = [t.replace(f"{prefix}_", "").replace("_raw", "") for t in matched_tables]
        return base_names

    def run_cleaning(self, snapshot):
        """
        Detects, cleans, and materializes all tables from the raw schema.
        """
        self.logger.info(f"🚀 Dynamically detecting tables in {self.raw_schema}...")

        # 1. Process Bank Mandiri Tables
        bm_sources = self._get_tables_by_pattern("bm")
        for source in bm_sources:
            try:
                df_clean = self.bm_handler.get_data(source, snapshot)
                target_table = f"{self.clean_schema}.bm_{source}"
                
                df_clean.write.format("delta").mode("overwrite") \
                        .option("replaceWhere", f"snapshot_date = '{snapshot}'") \
                        .saveAsTable(target_table)
                
                print(f"✅ BM Cleaned: {source} -> {target_table}")
            except Exception as e:
                print(f"❌ BM Failed for {source}: {str(e)}")

        # 2. Process AMFS Tables
        amfs_sources = self._get_tables_by_pattern("amfs")
        for source in amfs_sources:
            try:
                df_clean = self.amfs_handler.get_data(source, snapshot)
                target_table = f"{self.clean_schema}.amfs_{source}"
                
                df_clean.write.format("delta").mode("overwrite") \
                        .option("replaceWhere", f"snapshot_date = '{snapshot}'") \
                        .saveAsTable(target_table)
                
                print(f"✅ AMFS Cleaned: {source} -> {target_table}")
            except Exception as e:
                print(f"❌ AMFS Failed for {source}: {str(e)}")

        self.logger.info("--- Dynamic Handler Job Complete ---")