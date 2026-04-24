import logging
from pyspark.sql.functions import col, sum as _sum, when, lit, coalesce
from amfs_tp.features.bank.trx_lag import TrxLag

class TrxDebitEdc(TrxLag):
    def create_clean(self, snapshot):
        try:
            df = self.spark.table(f"{self.base_path}.debitcard_trx_edc_cif_cleaned") \
                     .filter(col("snapshot_date") == snapshot).fillna(0)
            
            if df.isEmpty(): return None

            group_maps = {
                'luxury': ['jewelry', 'antique'], 'health': ['hospital', 'medical'],
                'vehicle': ['spbu', 'vehiclepart', 'vehicle'], 'household': ['buildingmaterial', 'household', 'furniture'],
                'daily_life': ['fastfood', 'electrical', 'telco', 'utilities', 'transportation', 'othersprod', 'miscprod', 'mini_market', 'supermarket', 'deptstore'],
                'services': ['finservices', 'specialorg', 'profservices', 'eduserv', 'govserv'],
                'leisure_hobby': ['fashion', 'restaurant', 'bookstores', 'hobbies', 'electronic'],
                'travel': ['hotel', 'airlines'], 'offus': ['offus']
            }

            # 1. Group Summing Logic
            for key, categories in group_maps.items():
                f_cols = [f"{c}_freq" for c in categories if f"{c}_freq" in df.columns]
                s_cols = [f"{c}_sv" for c in categories if f"{c}_sv" in df.columns]
                
                sum_f = lit(0)
                for c in f_cols: sum_f = sum_f + coalesce(col(c), lit(0))
                df = df.withColumn(f"{key}_freq", sum_f)
                
                sum_s = lit(0.0)
                for c in s_cols: sum_s = sum_s + coalesce(col(c), lit(0.0))
                df = df.withColumn(f"{key}_sv", sum_s)

            # 2. Grand Totals
            all_f_cols = [f"{k}_freq" for k in group_maps.keys()]
            all_s_cols = [f"{k}_sv" for k in group_maps.keys()]
            
            total_f = lit(0)
            for c in all_f_cols: total_f = total_f + coalesce(col(c), lit(0))
            
            total_s = lit(0.0)
            for c in all_s_cols: total_s = total_s + coalesce(col(c), lit(0.0))

            df = df.withColumn("sum_freq", total_f).withColumn("sum_sv", total_s)

            # 3. Percentages
            for key in group_maps.keys():
                df = df.withColumn(f"{key}_sv_per", when(col("sum_sv") > 0, col(f"{key}_sv") / col("sum_sv")).otherwise(0.0))
            
            return df
        except Exception as e:
            logging.warning(f"TrxDebitEdc skipping {snapshot}: {str(e)}")
            return None

    def create(self, snapshot):
        """Generates engineered features for Debit EDC with legacy naming."""
        if self.df_raw is None: 
            self.create_raw(snapshot)
        
        df = self.df_raw
        
        # FIX: Changed 'returns' to 'return'
        if df is None: 
            return

        # 4. Legacy Lag Calculations
        df = self._diff_between(df, 'sv', 'sum_sv', 0, 2)
        df = self._diff_between(df, 'sv', 'sum_sv', 2, 5)
        df = self._diff_between(df, 'freq', 'sum_freq', 0, 2)
        df = self._diff_between(df, 'freq', 'sum_freq', 2, 5)
        
        # Passing sum_prefix=False to keep the 'max_sv_lm3' style naming
        df = self._basic_stats(df, x='sv', sum_prefix=False)
        df = self._basic_stats(df, x='freq', sum_prefix=False)

        # 5. Final Projection: Ensuring 'cifno' is not duplicated
        final_expr = [col("cifno")] + [
            col(c).alias(c.replace("_lag0", "")) if c.endswith("_lag0") else col(c) 
            for c in df.columns if c != "cifno" and ("_lag" not in c or c.endswith("_lag0"))
        ]
        
        self._save_to_fs(df.select(*final_expr), "debit_edc", snapshot)