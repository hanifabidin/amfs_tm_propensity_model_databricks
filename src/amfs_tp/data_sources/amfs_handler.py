import logging
import re
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

class AMFSHandler:
    def __init__(self, spark, catalog="amfs_tm"):
        self.spark = spark
        self.catalog = catalog
        self.raw_schema = f"{catalog}.raw"
        self.logger = logging.getLogger(__name__)
        
        self.id_cols = ["accnumber", "cifno", "account_no"]

    def _standardize(self, df):
        """Standardizes headers and casts IDs to String."""
        for c in df.columns:
            clean_name = re.sub(r'[ ,;{}()\n\t=]+', '_', c.lower().strip()).strip('_')
            df = df.withColumnRenamed(c, clean_name)

        # Standardize ID column for joins
        for id_col in self.id_cols:
            if id_col in df.columns:
                df = df.withColumn(id_col, F.col(id_col).cast(StringType()))
                
        return df

    def get_data(self, source_name, snapshot):
        """Fetch generic AMFS raw data (e.g. all_status, tso)."""
        raw_table = f"{self.raw_schema}.amfs_{source_name}_raw"
        df = self.spark.table(raw_table).filter(F.col("snapshot_date") == snapshot)
        
        if df.isEmpty():
            raise ValueError(f"AMFS Handler: Raw data missing for {source_name} in {snapshot}")
            
        return self._standardize(df)

    def get_call_labels(self, snapshot, mapping_df):
        """
        Creates the Training Target by joining Call Tracking with BM Mapping.
        mapping_df: Should be the output of BMHandler.get_data('cif_acct_mapping')
        """
        ct_df = self.get_data("call_tracking", snapshot)
        
        # Target Logic: Contacted (>=202), Converted (>=901)
        # We rename 'accnumber' to 'rekening' to match the mapping table join key
        labels = (ct_df.withColumn("callid_int", F.col("callid").cast("int"))
                    .filter(F.col("callid_int") >= 202)
                    .withColumn("label", F.when(F.col("callid_int") >= 901, 1).otherwise(0))
                    .withColumnRenamed("accnumber", "rekening_join")
                    .select("rekening_join", "label", "callid", "calldate"))
        
        # Final Join to return cifno
        # Assumes mapping_df has 'cifno' and 'rekening'
        return (labels.join(mapping_df, labels.rekening_join == mapping_df.cifno, how="inner")
                      .withColumn("snapshot_date", F.lit(snapshot))
                      .select(mapping_df.cifno, "label", "snapshot_date"))