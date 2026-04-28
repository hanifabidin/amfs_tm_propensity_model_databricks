from pyspark.sql import functions as F
import logging

class FilterEngine:
    def __init__(self, spark, catalog, config):
        self.spark = spark
        self.catalog = catalog
        self.config = config['rules']
        self.audit_log = []

    def _log_step(self, step_name, count, snapshot):
        self.audit_log.append({
            "step": step_name, 
            "cif_count": count, 
            "snapshot_date": snapshot
        })

    def apply_filters(self, snapshot):
        # 1. Base Population (from phsumm/bank_ph)
        rule_cfg = self.config['exclude_new_customer']
        df = self.spark.table(f"{self.catalog}.{rule_cfg['table']}") \
                       .filter(F.col("snapshot_date") == snapshot)
        
        # Rule 1: Vintage
        min_v = rule_cfg['params']['min_vint_months']
        df = df.filter(F.col("saving_acct_vint_l") >= min_v)
        self._log_step("01_vintage_filter", df.count(), snapshot)

        # Rule 2: Segment/Active Rule
        if self.config['segment_rule']['enabled']:
            seg_cfg = self.config['segment_rule']
            seg_df = self.spark.table(f"{self.catalog}.{seg_cfg['table']}") \
                               .filter(F.col("snapshot_date") == snapshot)
            
            # Left join and filter to keep only active/non-AXA
            df = df.join(seg_df, "cifno", "left")
            if seg_cfg['params'].get('exclude_non_active'):
                df = df.filter(F.col("bank_active_flag") == 1)
            if seg_cfg['params'].get('exclude_existing_axa'):
                df = df.filter(F.col("axa_pol_flag") == 0)
            
            self._log_step("02_segment_filter", df.count(), snapshot)

        # Save Audit Trail to Table
        audit_df = self.spark.createDataFrame(self.audit_log)
        audit_df.write.mode("append").saveAsTable(f"{self.catalog}.audit.filter_stats")
        
        return df.select("cifno", "snapshot_date")