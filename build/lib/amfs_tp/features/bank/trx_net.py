import logging
from pyspark.sql.functions import col
from amfs_tp.features.bank.trx_lag import TrxLag

class TrxNet(TrxLag):
    def create_clean(self, snapshot):
        try:
            # Using corrected table name
            df = self.spark.table(f"{self.base_path}.tran_net_cleaned") \
                     .filter(col("snapshot_date") == snapshot).fillna(0)
            
            # Standardization to cifno if necessary
            if "cifno" not in df.columns and "cif" in df.columns:
                df = df.withColumnRenamed("cif", "cifno")

            if "sum_cd" in df.columns and "sum_db" in df.columns:
                df = df.withColumn("sum_cd_db_diff", col("sum_cd") - col("sum_db"))
            if "num_cd" in df.columns and "num_db" in df.columns:
                df = df.withColumn("num_cd_db_diff", col("num_cd") - col("num_db"))
            return df
        except Exception as e:
            logging.warning(f"TrxNet.create_clean: Error processing {snapshot}: {str(e)}")
            return None

    def create_raw(self, snapshot):
        months = self._get_history_list(snapshot, 6)
        joined_df = None

        for i, m in enumerate(months):
            snap_df = self.create_clean(m)
            if snap_df is None: continue
            
            suffix = f"_lag{i}"
            use_cols = snap_df.columns if i == 0 else ['cifno', 'sum_db', 'sum_cd']
            snap_df = snap_df.select(*[c for c in use_cols if c in snap_df.columns])
            
            for c in snap_df.columns:
                if c != 'cifno':
                    snap_df = snap_df.withColumnRenamed(c, f"{c}{suffix}")
            
            joined_df = snap_df if joined_df is None else joined_df.join(snap_df, on="cifno", how="left")
        return joined_df

    def create(self, snapshot):
        if self.df_raw is None: self.create_raw(snapshot)
        df = self.df_raw
        if df is None: return

        df = self._diff_between(df, 'db', 'sum_db', 0, 2)
        df = self._diff_between(df, 'db', 'sum_db', 2, 5)
        df = self._diff_between(df, 'cd', 'sum_cd', 0, 2)
        df = self._diff_between(df, 'cd', 'sum_cd', 2, 5)
        df = self._basic_stats(df, 'db')
        df = self._basic_stats(df, 'cd')

        # FIX: Skip 'cifno' in the list comprehension
        final_expr = [col("cifno")] + [
            col(c).alias(c.replace("_lag0", "")) if c.endswith("_lag0") else col(c) 
            for c in df.columns if c != "cifno" and ("_lag" not in c or c.endswith("_lag0"))
        ]

        self._save_to_fs(df.select(*final_expr), "trx_net", snapshot)