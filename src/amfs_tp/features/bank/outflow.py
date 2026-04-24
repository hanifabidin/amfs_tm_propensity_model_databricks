import logging
from pyspark.sql.functions import col, sum as _sum, when
from amfs_tp.features.bank.trx_lag import TrxLag

class TrxOutflow(TrxLag):
    def create_clean(self, snapshot):
        try:
            df = self.spark.table(f"{self.base_path}.trx_outflow_cleaned") \
                     .filter(col("snapshot_date") == snapshot) \
                     .withColumnRenamed("cifno_pengirim", "cifno") \
                     .fillna(0)
            
            if df.isEmpty(): 
                return None

            agg_cols = [c for c in df.columns if c not in ['cifno', 'rekening_pengirim', 'snapshot_date']]
            df = df.groupby("cifno").agg(*[_sum(c).alias(c) for c in agg_cols])

            # Legacy percentage logic
            for c in ['withdrawal', 'transfer_to_mandiri', 'transfer_to_others', 'bill_payment', 'trx_others']:
                if c in df.columns:
                    df = df.withColumn(f"{c}_per", when(col("trx_total") > 0, col(c) / col("trx_total")).otherwise(0.0))
            return df
        except Exception as e:
            logging.warning(f"TrxOutflow.create_clean: Snapshot {snapshot} failed: {str(e)}")
            return None

    def create(self, snapshot):
        """Generates engineered features. Resilient to missing history."""
        if self.df_raw is None: 
            self.create_raw(snapshot)
        
        df = self.df_raw
        
        # --- CRITICAL GUARD ---
        if df is None:
            logging.error(f"TrxOutflow Execution Aborted: No valid joined data found for {snapshot}.")
            return

        # re-assigning to 'df' ensures the object reference is never lost
        df = self._diff_between(df, 'trxout', 'trx_total', 0, 2)
        df = self._diff_between(df, 'trxout', 'trx_total', 2, 5)
        df = self._basic_stats(df, 'trxout', sum_prefix=False)

        # FIX: Skip 'cifno' in the list comprehension
        final_expr = [col("cifno")] + [
            col(c).alias(c.replace("_lag0", "")) if c.endswith("_lag0") else col(c) 
            for c in df.columns if c != "cifno" and ("_lag" not in c or c.endswith("_lag0"))
        ]
        # The .select() call is now safe because df is guaranteed to be a DataFrame here
        self._save_to_fs(df.select(*final_expr), "trx_outflow", snapshot)