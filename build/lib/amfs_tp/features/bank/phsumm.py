from pyspark.sql.functions import (
    col, lit, when, to_date, datediff, last_day, 
    coalesce, sum as _sum, least, greatest, expr
)
from amfs_tp.features.bank.balance import Balance

class BankPH(Balance):
    def create_clean(self, snapshot):
        """Standardizes PHSUMM: Capping, Date Validation, and Vintages."""
        # 1. Load and Initial Mapping
        df = self.spark.table(f"{self.base_path}.phsumm_cleaned") \
                 .filter(col("snapshot_date") == snapshot) \
                 .withColumnRenamed("loanmort_s", "loanmort") \
                 .withColumnRenamed("loanall_s", "loanall")
        
        # 2. Filter: Exclude rows where only T12 (Insurance) exists
        if "totph" in df.columns and "t12" in df.columns:
            df = df.filter(col("totph") != col("t12"))

        # 3. Handle Numerical (Capping & Totals)
        ph_count_cols = [f"t{i}" for i in range(1, 19)]
        for c in ph_count_cols:
            if c in df.columns:
                # Cap t1-t18 at 5
                df = df.withColumn(c, when(col(c) > 5, 5).otherwise(coalesce(col(c), lit(0))))

        df = df.withColumn("totph", when(col("totph") > 15, 15).otherwise(coalesce(col("totph"), lit(15))))
        
        # TOT_SAVING: Sum t1-t9, capped at 11
        s_cols = [f"t{i}" for i in range(1, 10) if f"t{i}" in df.columns]
        df = df.withColumn("tot_saving", least(lit(11), sum([col(c) for c in s_cols])))
        
        # TOT_LOAN: Sum t12-t17, capped at 4
        l_cols = [f"t{i}" for i in range(12, 18) if f"t{i}" in df.columns]
        df = df.withColumn("tot_loan", least(lit(4), sum([col(c) for c in l_cols])))

        # 4. Handle Dates & Vintages
        # Note: If hardening cast these to DOUBLE, we cast to STRING then DATE
        date_cols = ['sfstopn', 'slstopn', 'dfstopn', 'dlstopn', 'gfstopn', 'glstopn', 'cfstopn', 'clstopn']
        ref_date = last_day(to_date(lit(snapshot), 'yyyyMM'))
        
        for c in date_cols:
            if c in df.columns:
                # Convert numeric YYYYMMDD to Date
                df = df.withColumn(c, to_date(col(c).cast("string"), "yyyy-MM-dd"))
                # Legacy outlier logic: 1917-1980 becomes 1980
                df = df.withColumn(c, when((col(c) < "1980-01-01") & (col(c) > "1917-01-01"), to_date(lit("1980-01-01")))
                                     .when(col(c) > ref_date, lit(None))
                                     .otherwise(col(c)))

        # Calculate First and Last across all types
        f_cols = [c for c in ['sfstopn', 'dfstopn', 'gfstopn', 'cfstopn'] if c in df.columns]
        l_cols = [c for c in ['slstopn', 'dlstopn', 'glstopn', 'clstopn'] if c in df.columns]
        
        if f_cols: df = df.withColumn("all_accts_first_date", least(*f_cols))
        if l_cols: df = df.withColumn("all_accts_last_date", greatest(*l_cols))

        # Vintage calculation (Months = days / 30.44)
        vint_map = {
            'saving_acct_vint_l': 'slstopn',
            'saving_acct_vint_f': 'sfstopn',
            'all_acct_vint_l': 'all_accts_last_date',
            'all_acct_vint_f': 'all_accts_first_date'
        }
        
        for feat, source in vint_map.items():
            if source in df.columns:
                df = df.withColumn(feat, datediff(ref_date, col(source)) / 30.44)

        # Drop raw date columns
        df = df.drop(*(date_cols + ['all_accts_first_date', 'all_accts_last_date']))
        return df

    def create(self, snapshot):
        """Main PHSUMM feature creation."""
        df = self.create_clean(snapshot)
        self._save_to_fs(df, "bank_ph", snapshot)

    def create_loan(self, snapshot):
        """Temporal logic for Loan Increments using a 4-month window."""
        # Relative lags for t, t-1, t-2, t-3 (4 months total)
        months = self._get_history_list(snapshot, 4)
        joined_df = None
        
        for i, m in enumerate(months):
            try:
                snap_df = self.spark.table(f"{self.base_path}.phsumm_cleaned") \
                              .filter(col("snapshot_date") == m) \
                              .select("cifno", "loanall_s") \
                              .withColumnRenamed("loanall_s", f"loanall_lag{i}")
                
                joined_df = snap_df if joined_df is None else joined_df.join(snap_df, on="cifno", how="left")
            except: continue

        if not joined_df: return
        
        # Calculate flags (lag_{i} > lag_{i+1} means an increase occurred)
        for i in range(3):
            curr, prev = f"loanall_lag{i}", f"loanall_lag{i+1}"
            if curr in joined_df.columns and prev in joined_df.columns:
                joined_df = joined_df.withColumn(f"loan_inc_flag_{i}", 
                                                 when(col(curr) > col(prev), 1).otherwise(0))
        
        flag_cols = [f"loan_inc_flag_{i}" for i in range(3) if f"loan_inc_flag_{i}" in joined_df.columns]
        joined_df = joined_df.withColumn("loan_recent_inc_count", sum([col(c) for c in flag_cols]))
        
        # Loan to Debit estimate (8% of current loan)
        if "loanall_lag0" in joined_df.columns:
            joined_df = joined_df.withColumn("loan_to_debit", col("loanall_lag0") * 0.08)
        
        # Keep only engineered features
        keep_cols = ["cifno", "loan_recent_inc_count", "loan_to_debit"] + [f"loan_inc_flag_{i}" for i in range(3)]
        self._save_to_fs(joined_df.select(*[c for c in keep_cols if c in joined_df.columns]), "loan_diff", snapshot)

    def create_saving_account(self, snapshot):
        """Calculates Undebit-to-Total Saving ratios."""
        df = self.spark.table(f"{self.base_path}.phsumm_cleaned") \
                 .filter(col("snapshot_date") == snapshot)
        
        s_cols = [f't{i}' for i in range(1, 10) if f't{i}' in df.columns]
        u_cols = [f't{i}' for i in [3, 4, 9] if f't{i}' in df.columns]
        
        df = df.withColumn("tot_saving", sum([col(c) for c in s_cols])) \
               .withColumn("tot_undebit_saving", sum([col(c) for c in u_cols]))
        
        df = df.withColumn("ratio_undebit_saving", 
                           coalesce(col("tot_undebit_saving") / when(col("tot_saving") == 0, lit(None)).otherwise(col("tot_saving")), lit(1.0)))
        
        self._save_to_fs(df.select("cifno", "tot_saving", "tot_undebit_saving", "ratio_undebit_saving"), "saving_acct", snapshot)