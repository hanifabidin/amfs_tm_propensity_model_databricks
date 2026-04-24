import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, to_date, datediff, date_format, 
    dayofmonth, greatest, least, coalesce, sum as _sum
)
from databricks.feature_engineering import FeatureEngineeringClient

class Balance:
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        self.spark = spark
        self.fe = FeatureEngineeringClient()
        self.catalog = catalog
        self.schema = schema
        self.base_path = f"{catalog}.{schema}"
        self.df_cleaned = None
        self.df_raw = None

    def _get_history_list(self, snapshot, period=6):
        start_dt = datetime.strptime(snapshot, '%Y%m')
        return [(start_dt - relativedelta(months=i)).strftime('%Y%m') for i in range(period)]

    def create_clean(self, snapshot):
        """Standardizes dates, clips ranges, and creates bins."""
        snap_dt = datetime.strptime(snapshot, '%Y%m')
        min_date = snap_dt - relativedelta(days=1)
        max_date = snap_dt + relativedelta(months=1)

        df = self.spark.table(f"{self.base_path}.cifsumm_cleaned") \
                 .filter(col("snapshot_date") == snapshot)

        for c in ['mxdt', 'mndt']:
            if c in df.columns:
                df = df.withColumn(c, to_date(col(c))) \
                       .withColumn(c, when((col(c) < lit(min_date)) | (col(c) > lit(max_date)), None).otherwise(col(c))) \
                       .withColumn(f"weekday_{c}", date_format(col(c), 'EEEE'))
                
                day = dayofmonth(col(c))
                df = df.withColumn(f"{c}_range", 
                        when(day.between(1, 10), 1).when(day.between(11, 20), 2)
                        .when(day > 20, 3).otherwise(-1))

        if 'mxdt' in df.columns and 'mndt' in df.columns:
            df = df.withColumn('delta_days', datediff(col('mxdt'), col('mndt')))
        if 'mx' in df.columns and 'mn' in df.columns:
            df = df.withColumn('delta_max_min_bal', col('mx') - col('mn'))

        self.df_cleaned = df
        return df

    def create_raw(self, snapshot):
        """Joins 6 months of historical data using relative lag suffixes."""
        months = self._get_history_list(snapshot, 6)
        base_cols = [
            'cifno', 'avg', 'sdev', 'med', 'mx', 'mn', 'qt1', 'qt3', 
            'd20', 'eom', 'nod', 'delta_days', 'mxdt_range', 
            'mndt_range', 'delta_max_min_bal'
        ]

        joined_df = None
        for i, m in enumerate(months):
            # 1. Call create_clean for the specific month
            snap_df = self.create_clean(m)
            
            # --- COLD START GUARD ---
            if snap_df is None:
                import logging
                logging.warning(f"{self.__class__.__name__}: History missing for {m}. Skipping.")
                continue
            
            # Using relative suffixes (lag0, lag1...) instead of hardcoded dates
            suffix = f"_lag{i}"
            
            # For the current month (lag0), we might want to keep weekday columns
            use_cols = base_cols + (['weekday_mxdt', 'weekday_mndt'] if i == 0 else [])
            # For older months (3+), we often only need the main balance points
            if i >= 3: use_cols = ['cifno', 'avg', 'mx', 'mn', 'eom']
            
            snap_df = snap_df.select([c for c in use_cols if c in snap_df.columns])
            
            # Rename columns to include the lag suffix
            for c in snap_df.columns:
                if c != 'cifno':
                    snap_df = snap_df.withColumnRenamed(c, f"{c}{suffix}")
            
            if joined_df is None:
                joined_df = snap_df
            else:
                joined_df = joined_df.join(snap_df, on="cifno", how="left")

        self.df_raw = joined_df
        return joined_df

    def create(self, snapshot):
        """
        Generates 3-month and 6-month windowed stats.
        Includes internal type-casting to prevent 'String vs Double' merge errors.
        """
        if self.df_raw is None: self.create_raw(snapshot)
        df = self.df_raw

        for window in [3, 6]:
            suffix = f"{window}mth"
            
            # Use internal casting to ensure all columns are Double before comparing
            mx_cols = [col(f"mx_lag{i}").cast("double") for i in range(window) if f"mx_lag{i}" in df.columns]
            mn_cols = [col(f"mn_lag{i}").cast("double") for i in range(window) if f"mn_lag{i}" in df.columns]
            avg_cols = [col(f"avg_lag{i}").cast("double") for i in range(window) if f"avg_lag{i}" in df.columns]

            # MAX Logic
            if len(mx_cols) > 1:
                df = df.withColumn(f"max_bal_{suffix}", greatest(*mx_cols))
            elif len(mx_cols) == 1:
                df = df.withColumn(f"max_bal_{suffix}", mx_cols[0])

            # MIN Logic (least() skips nulls automatically)
            if len(mn_cols) > 1:
                df = df.withColumn(f"min_bal_{suffix}", least(*mn_cols))
            elif len(mn_cols) == 1:
                df = df.withColumn(f"min_bal_{suffix}", mn_cols[0])

            # AVG Logic (Handles partial history)
            if avg_cols:
                sum_val = sum([coalesce(c, lit(0)) for c in avg_cols])
                count_non_nulls = sum([when(c.isNotNull(), 1).otherwise(0) for c in avg_cols])
                
                df = df.withColumn(f"avg_bal_{suffix}", 
                                   sum_val / when(count_non_nulls == 0, 1).otherwise(count_non_nulls))

        # 2. Momentum Features (t1-2 and t2-3)
        for i in range(2):
            t_curr, t_prev = f"avg_lag{i}", f"avg_lag{i+1}"
            tag = f"t{i+1}-{i+2}"
            if t_curr in df.columns and t_prev in df.columns:
                # Ensure these are doubles too
                c_curr = col(t_curr).cast("double")
                c_prev = col(t_prev).cast("double")
                
                diff_col = f"delta_avg_bal_{tag}"
                df = df.withColumn(diff_col, c_curr - c_prev) \
                       .withColumn(f"ratio_avg_bal_{tag}", 
                                   col(diff_col) / when(c_prev == 0, lit(None)).otherwise(c_prev))

        # 3. Final Selection: Rename lag0 and keep Engineered Features
        final_expressions = []
        for c in df.columns:
            if c.endswith("_lag0"):
                clean_name = c.replace("_lag0", "")
                final_expressions.append(col(c).alias(clean_name))
            elif any(x in c for x in ["_3mth", "_6mth", "t1-2", "t2-3"]) or c == "cifno":
                final_expressions.append(col(c))

        df_final = df.select(*final_expressions)
        self._save_to_fs(df_final, "balance_window", snapshot)

    def create_real(self, snapshot):
        """Final loan-deducted table. Skips non-numeric columns like weekdays."""
        try:
            loan_df = self.spark.table(f"{self.base_path}.loan_diff_features") \
                          .filter(col("snapshot_date") == snapshot) \
                          .select("cifno", "loan_to_debit")
            
            df = self.df_raw if self.df_raw else self.spark.table(f"{self.base_path}.balance_window_features").filter(col("snapshot_date") == snapshot)
            df = df.join(loan_df, on="cifno", how="left").fillna(0)
            
            # Numeric targets only
            target_keywords = ['avg', 'mx', 'mn', 'eom', 'max_bal', 'avg_bal']
            
            for c in df.columns:
                # FIX: Added 'weekday', 'range', and 'date' to the exclusion list
                is_target = any(k in c for k in target_keywords)
                is_numeric = all(x not in c for x in ['ratio', 'cifno', 'weekday', 'range', 'date', 'snapshot'])
                
                if is_target and is_numeric:
                    df = df.withColumn(f"{c}_loan", col(c) - col("loan_to_debit"))
            
            self._save_to_fs(df.drop("loan_to_debit"), "balance_loan", snapshot)
        except Exception as e: 
            logging.warning(f"Skipping create_real: {str(e)}")

    def create_deduct(self, snapshot):
        """Deducts loan from raw D-level columns."""
        try:
            loan_df = self.spark.table(f"{self.base_path}.loan_diff_features") \
                          .filter(col("snapshot_date") == snapshot) \
                          .select("cifno", "loan_to_debit")
            
            df = self.spark.table(f"{self.base_path}.cifsumm_cleaned") \
                     .filter(col("snapshot_date") == snapshot) \
                     .select("cifno", "avgd", "mnd", "eomd") \
                     .join(loan_df, on="cifno", how="left") \
                     .fillna(0)
            
            for c in ["avgd", "mnd", "eomd"]:
                df = df.withColumn(f"{c}_loan", col(c) - col("loan_to_debit"))
            
            self._save_to_fs(df.drop("loan_to_debit"), "balance_deduct", snapshot)
        except Exception as e:
            import logging
            logging.warning(f"Skipping create_deduct: {str(e)}")

    def _save_to_fs(self, df, table_name, snapshot):
        full_name = f"{self.base_path}.{table_name}_features"
        df = df.withColumn("snapshot_date", lit(snapshot))
        try:
            self.fe.write_table(name=full_name, df=df, mode="merge")
        except:
            self.fe.create_table(name=full_name, primary_keys=["cifno"], df=df)