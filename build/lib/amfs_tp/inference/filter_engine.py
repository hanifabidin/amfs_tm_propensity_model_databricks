from pyspark.sql import functions as F
import logging

class FilterEngine:
    def __init__(self, spark, catalog, config):
        self.spark = spark
        self.catalog = catalog
        self.config = config['rules']
        self.audit_log = []
        self.logger = logging.getLogger(__name__)

    def _log_step(self, step_name, count, snapshot):
        self.audit_log.append((step_name, count, snapshot))

    def apply_filters(self, snapshot):
        self.logger.info(f"Applying filters for snapshot {snapshot}")

        # 0. MASTER POPULATION (The Universe)
        # We start with the mapping table which is guaranteed to have the full list of CIFs
        base_table = f"{self.catalog}.clean.bm_cif_acct_mapping"
        
        if not self.spark.catalog.tableExists(base_table):
            raise FileNotFoundError(f"Critical Error: Master mapping table {base_table} not found.")

        # Start with all unique CIFs for this month
        df = self.spark.table(base_table) \
                       .filter(F.col("snapshot_date") == snapshot) \
                       .select("cifno", "snapshot_date") \
                       .distinct()
        
        initial_count = df.count()
        self._log_step("00_master_population", initial_count, snapshot)
        
        if initial_count == 0:
            self.logger.error(f"❌ Master population is empty for {snapshot}. Pipeline stopped.")
            return df

        # 1. Iterate through rules in config
        for rule_name, rule_cfg in self.config.items():
            if not rule_cfg.get('enabled'):
                continue
            
            full_table_path = f"{self.catalog}.{rule_cfg['table']}"
            
            # --- THE GUARD: Skip rule if the feature table is missing or empty ---
            if not self.spark.catalog.tableExists(full_table_path):
                self.logger.warning(f"⚠️ Skipping rule '{rule_name}': Table {full_table_path} does not exist.")
                continue
            
            rule_df = self.spark.table(full_table_path).filter(F.col("snapshot_date") == snapshot)
            
            if rule_df.isEmpty():
                self.logger.warning(f"⚠️ Skipping rule '{rule_name}': Table {full_table_path} is empty for {snapshot}.")
                continue

            # Join the feature needed for the rule
            df = df.join(rule_df, rule_cfg['key'], "inner")
            
            # Apply specific logic for 'exclude_new_customer' (Vintage)
            if rule_name == 'exclude_new_customer':
                min_v = rule_cfg['params']['min_vint_months']
                if 'saving_acct_vint_l' in df.columns:
                    df = df.filter(F.col("saving_acct_vint_l") >= min_v)
            
            # Apply specific logic for 'segment_rule'
            elif rule_name == 'segment_rule':
                if rule_cfg['params'].get('exclude_non_active'):
                    df = df.filter(F.col("bank_active_flag") != 0)
                if rule_cfg['params'].get('exclude_existing_axa'):
                    df = df.filter(F.col("axa_pol_flag") != 1)

            self._log_step(f"step_{rule_name}", df.count(), snapshot)

        # Save Audit Trail to the audit schema
        audit_table = f"{self.catalog}.audit.filter_stats"
        audit_df = self.spark.createDataFrame(self.audit_log, ["step", "cif_count", "snapshot_date"])
        audit_df.write.mode("append").saveAsTable(audit_table)
        
        return df.select("cifno", "snapshot_date")