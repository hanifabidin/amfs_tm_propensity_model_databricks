import logging
from pyspark.sql.functions import col, lit, when, coalesce, greatest, least
from amfs_tp.features.bank.balance import Balance

class TrxLag(Balance):
    def _diff_between(self, df, newcol_pref, usecol_pref, start_lag, end_lag):
        """Chain-safe legacy diff logic: always returns a DataFrame."""
        if df is None: 
            return None
        
        l_map = {0: 'lm', 1: 'lm2', 2: 'lm3', 3: 'lm4', 4: 'lm5', 5: 'lm6'}
        start_tag, end_tag = l_map[start_lag], l_map[end_lag]
        new_col = f"{newcol_pref}_{start_tag}_{end_tag}_diff"
        
        c_recent_name = f"{usecol_pref}_lag{start_lag}"
        c_past_name = f"{usecol_pref}_lag{end_lag}"

        # COLD START DEFENSE: If history columns are missing, return DF as-is with a 0.0 column
        if c_recent_name not in df.columns or c_past_name not in df.columns:
            logging.warning(f"Feature {new_col} skipped: {c_recent_name} or {c_past_name} not found.")
            return df.withColumn(new_col, lit(0.0))

        c_recent = col(c_recent_name).cast("double")
        c_past = col(c_past_name).cast("double")

        # Formula: $$\Delta = \frac{Recent - Past}{Past}$$
        return df.withColumn(new_col, 
            when(c_past > 0, (c_recent - c_past) / c_past)
            .when((c_past == 0) & (c_recent != 0), c_recent)
            .otherwise(0.0)
        )

    def _basic_stats(self, df, x, sum_prefix=True):
        """Chain-safe legacy stats: ensures DF is returned even if windows are empty."""
        if df is None: 
            return None
        
        for window_size in [3, 6]:
            tag = f"lm{window_size}"
            cols = [col(f"{x}_lag{i}").cast("double") for i in range(window_size) if f"{x}_lag{i}" in df.columns]
            
            if len(cols) > 0:
                # Max/Min logic
                if len(cols) > 1:
                    df = df.withColumn(f"max_{x}_{tag}", greatest(*cols)) \
                           .withColumn(f"min_{x}_{tag}", least(*cols))
                else:
                    df = df.withColumn(f"max_{x}_{tag}", cols[0]) \
                           .withColumn(f"min_{x}_{tag}", cols[0])
                
                # Mean logic (Sum / Count)
                sum_expr = sum([coalesce(c, lit(0)) for c in cols])
                count_expr = sum([when(c.isNotNull(), 1).otherwise(0) for c in cols])
                df = df.withColumn(f"mean_{x}_{tag}", sum_expr / when(count_expr == 0, 1).otherwise(count_expr))
            else:
                # Placeholder nulls for model stability
                df = df.withColumn(f"max_{x}_{tag}", lit(None).cast("double")) \
                       .withColumn(f"min_{x}_{tag}", lit(None).cast("double")) \
                       .withColumn(f"mean_{x}_{tag}", lit(None).cast("double"))
                       
        return df