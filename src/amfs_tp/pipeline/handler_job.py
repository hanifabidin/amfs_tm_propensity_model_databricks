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
        
        self.bm_handler = BMHandler(spark, catalog=catalog)
        self.amfs_handler = AMFSHandler(spark, catalog=catalog)

    def _get_tables_by_pattern(self, prefix):
        tables = self.spark.catalog.listTables(self.raw_schema)
        matched_tables = [
            t.name for t in tables 
            if t.name.startswith(f"{prefix}_") and t.name.endswith("_raw")
        ]
        base_names = [t.replace(f"{prefix}_", "").replace("_raw", "") for t in matched_tables]
        return base_names

    def run_cleaning(self, snapshot):
        self.logger.info(f"🚀 Dynamically detecting tables in {self.raw_schema}...")

        # 1. Process Bank Mandiri Tables
        bm_sources = self._get_tables_by_pattern("bm")
        for source in bm_sources:
            try:
                df_clean = self.bm_handler.get_data(source, snapshot)
                
                # New Naming Convention (Physical Table)
                new_table_name = f"{self.clean_schema}.bm_{source}"
                df_clean.write.format("delta").mode("overwrite") \
                        .option("replaceWhere", f"snapshot_date = '{snapshot}'") \
                        .saveAsTable(new_table_name)
                
                # --- LEGACY BRIDGE: Create View for Legacy Feature Classes ---
                # This maps 'phsumm_cleaned' -> 'bm_phsumm'
                legacy_view_name = f"{self.clean_schema}.{source}_cleaned"
                self.spark.sql(f"CREATE OR REPLACE VIEW {legacy_view_name} AS SELECT * FROM {new_table_name}")
                
                print(f"✅ BM Cleaned: {source} (Physical: {new_table_name}, View: {legacy_view_name})")
            except Exception as e:
                print(f"❌ BM Failed for {source}: {str(e)}")

        # 2. Process AMFS Tables
        amfs_sources = self._get_tables_by_pattern("amfs")
        for source in amfs_sources:
            try:
                df_clean = self.amfs_handler.get_data(source, snapshot)
                
                # New Naming Convention
                new_table_name = f"{self.clean_schema}.amfs_{source}"
                df_clean.write.format("delta").mode("overwrite") \
                        .option("replaceWhere", f"snapshot_date = '{snapshot}'") \
                        .saveAsTable(new_table_name)

                # --- LEGACY BRIDGE for AMFS ---
                legacy_view_name = f"{self.clean_schema}.{source}_cleaned"
                self.spark.sql(f"CREATE OR REPLACE VIEW {legacy_view_name} AS SELECT * FROM {new_table_name}")
                
                print(f"✅ AMFS Cleaned: {source} (Physical: {new_table_name}, View: {legacy_view_name})")
            except Exception as e:
                print(f"❌ AMFS Failed for {source}: {str(e)}")

        self.logger.info("--- Dynamic Handler Job with Legacy Bridging Complete ---")