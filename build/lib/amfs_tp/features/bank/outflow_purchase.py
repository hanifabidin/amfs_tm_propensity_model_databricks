import logging
from pyspark.sql.functions import col, sum as _sum, when, lit, coalesce
from amfs_tp.features.bank.trx_lag import TrxLag

class TrxOutflowPayment(TrxLag):
    def create_clean(self, snapshot):
        try:
            df = self.spark.table(f"{self.base_path}.trx_outflow_payment_purchase_cleaned") \
                     .filter(col("snapshot_date") == snapshot) \
                     .withColumnRenamed("customer_no", "cifno")
            
            if df.isEmpty(): return None

            # 1. Pivot Logic
            df_pivoted = df.groupBy("cifno").pivot("transaction_type").agg(
                _sum("frek").alias("freq"), _sum("nominal").alias("amt")
            ).fillna(0)

            # 2. Correct Horizontal Summing (Safe from Cold Start)
            freq_cols = [c for c in df_pivoted.columns if "_freq" in c]
            amt_cols = [c for c in df_pivoted.columns if "_amt" in c]
            
            # Initialize sums as literals to avoid [NOT_COLUMN] error
            sum_freq = lit(0)
            for c in freq_cols: sum_freq = sum_freq + coalesce(col(c), lit(0))
            
            sum_amt = lit(0.0)
            for c in amt_cols: sum_amt = sum_amt + coalesce(col(c), lit(0.0))

            df_pivoted = df_pivoted.withColumn("sum_purchase_freq", sum_freq) \
                                   .withColumn("sum_purchase_amt", sum_amt)

            # 3. Legacy Percentage Distribution
            for c in amt_cols:
                df_pivoted = df_pivoted.withColumn(f"{c}_per", 
                    when(col("sum_purchase_amt") > 0, col(c).cast("double") / col("sum_purchase_amt"))
                    .otherwise(0.0))
            return df_pivoted
        except Exception as e:
            logging.warning(f"TrxOutflowPayment skipping {snapshot}: {str(e)}")
            return None

    def create(self, snapshot):
        if self.df_raw is None: 
            self.df_raw = self.create_raw(snapshot)
        
        df = self.df_raw
        if df is None: return

        # Apply legacy lag logic
        df = self._diff_between(df, 'purchase_amt', 'sum_purchase_amt', 0, 2)
        df = self._diff_between(df, 'purchase_amt', 'sum_purchase_amt', 2, 5)
        df = self._basic_stats(df, 'purchase_amt', sum_prefix=False)

        # FIX: Filter 'cifno' out of the comprehension to avoid [COLUMN_ALREADY_EXISTS]
        final_expr = [col("cifno")] + [
            col(c).alias(c.replace("_lag0", "")) if c.endswith("_lag0") else col(c) 
            for c in df.columns if c != "cifno" and ("_lag" not in c or c.endswith("_lag0"))
        ]

        self._save_to_fs(df.select(*final_expr), "trx_outflow_purchase", snapshot)