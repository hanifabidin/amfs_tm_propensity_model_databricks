import logging
import re
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

class BMHandler:
    def __init__(self, spark, catalog="amfs_tm"):
        self.spark = spark
        self.catalog = catalog
        self.raw_schema = f"{catalog}.raw"
        self.clean_schema = f"{catalog}.clean"
        self.logger = logging.getLogger(__name__)

        # ID Mapping from your config
        self.id_cols = ["rekening", "cusno", "cifno_15", "cifno_01", "cifno_pengirim"]
        
        # Protected columns from legacy SparkCleaner
        self.protected = [
            'cifno', 'acct_no', 'accnumber', 'calldate', 'type', 'id', 'zipcode', 
            'status', 'date', 'periodp_id', 'cusno', 'branch_id', 'snapshot_date'
        ]

    def _validate_and_clean(self, df):
        """Sanitizes names, trims strings, and maps ID to 'cifno'."""
        # 1. Column Name Sanitization
        for c in df.columns:
            clean_name = re.sub(r'[ ,;{}()\n\t=]+', '_', c.lower().strip()).strip('_')
            df = df.withColumnRenamed(c, clean_name)

        # 2. Base String Cleanup (Trims and Nulls)
        for field in df.schema.fields:
            if isinstance(field.dataType, StringType):
                c = field.name
                df = (df.withColumn(c, F.regexp_replace(F.col(c), r'\b00:00:00\b', ''))
                        .withColumn(c, F.when(F.col(c) == '(null)', F.lit(None)).otherwise(F.col(c)))
                        .withColumn(c, F.trim(F.col(c))))

        # 3. ID Standardization
        potential_ids = [c for c in df.columns if c in self.id_cols]
        if potential_ids and 'cifno' not in df.columns:
            df = df.withColumn('cifno', F.coalesce(*[F.col(c) for c in potential_ids]))
        
        if 'cifno' in df.columns:
            df = df.withColumn('cifno', F.col('cifno').cast(StringType()))
            
        return df

    def _harden_types(self, df):
        """Casts non-ID/Date columns to Double for ML math."""
        for c in df.columns:
            is_protected = c.lower() in self.protected or \
                           any(c.lower().endswith(p) for p in ['_id', '_fg', '_date'])
            if not is_protected:
                df = df.withColumn(c, F.expr(f"try_cast({c} as double)"))
        return df

    def get_data(self, source_name, snapshot):
        """
        Pulls data from amfs_tm.raw, cleans it, and returns the 'clean' version.
        Example: handler.get_data("phsumm", "202505")
        """
        raw_table = f"{self.raw_schema}.bm_{source_name}_raw"
        self.logger.info(f"Handler: Processing {raw_table} for snapshot {snapshot}")

        # Partition Pruning for cost efficiency
        df = self.spark.table(raw_table).filter(F.col("snapshot_date") == snapshot)
        
        if df.isEmpty():
            raise ValueError(f"CRITICAL: Raw BM data not found for {source_name} in {snapshot}")

        df_cleaned = self._validate_and_clean(df)
        df_final = self._harden_types(df_cleaned)

        return df_final