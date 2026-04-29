import pandas as pd
import numpy as np
import yaml
import logging
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

class MatrixEngine:
    def __init__(self, spark: SparkSession, catalog: str, matrix_config: dict, impute_cfg_path: str):
        """
        Handles merging of feature sets and performs imputation/dummification.
        
        :param spark: Active SparkSession
        :param catalog: Unity Catalog name (e.g., 'amfs_tm')
        :param matrix_config: Dictionary loaded from matrix_config.yaml
        :param impute_cfg_path: Path to the impute_config.yaml file
        """
        self.spark = spark
        self.catalog = catalog
        self.matrix_cfg = matrix_config
        self.logger = logging.getLogger(__name__)
        
        # Load the imputation rules from the config directory
        try:
            with open(impute_cfg_path, 'r') as f:
                self.impute_cfg = yaml.safe_load(f)
            self.logger.info(f"Successfully loaded imputation config from {impute_cfg_path}")
        except Exception as e:
            self.logger.error(f"Failed to load imputation config: {str(e)}")
            raise

    def create_and_impute(self, population_df, snapshot: str):
        """
        Main execution flow for matrix preparation.
        """
        self.logger.info(f"Starting Matrix Generation for snapshot: {snapshot}")

        # --- PHASE 1: SPARK-BASED FEATURE MERGING ---
        # Starts with the filtered CIF population and joins feature tables
        matrix = population_df
        
        feature_sets = self.matrix_cfg.get('feature_sets', {})
        for name, conf in feature_sets.items():
            if not conf.get('enabled'):
                self.logger.info(f"Skipping feature set '{name}' (disabled in config).")
                continue
            
            full_table_path = f"{self.catalog}.{conf['table']}"
            
            # GUARD: Check if table exists in Unity Catalog before attempting join
            if not self.spark.catalog.tableExists(full_table_path):
                self.logger.warning(f"⚠️ TABLE NOT FOUND: {full_table_path}. Skipping these features.")
                continue
            
            self.logger.info(f"Merging feature set: {name} from {full_table_path}")
            
            # Load and filter by snapshot
            feat_df = self.spark.table(full_table_path) \
                                .filter(F.col("snapshot_date") == snapshot) \
                                .drop("snapshot_date")
            
            # Protection against [COLUMN_ALREADY_EXISTS] errors
            dup_cols = [c for c in feat_df.columns if c in matrix.columns and c != "cifno"]
            if dup_cols:
                self.logger.debug(f"Dropping duplicate columns in {name}: {dup_cols}")
                feat_df = feat_df.drop(*dup_cols)
            
            matrix = matrix.join(feat_df, on="cifno", how="left")

        # --- PHASE 2: PANDAS-BASED POST-PROCESSING ---
        # Convert to Pandas for detailed imputation and dummification logic
        self.logger.info("Converting merged matrix to Pandas for imputation and dummification...")
        df = matrix.toPandas()
        
        # Standardize column casing and spacing
        df.columns = df.columns.str.lower().str.strip()

        # 1. Vintage Preprocessing (Calculates months since join)
        snapshot_dt = pd.to_datetime(f"{snapshot[:4]}-{snapshot[4:]}-01")
        if 'date_org' in df.columns:
            df['date_org'] = pd.to_datetime(df['date_org'], dayfirst=True, errors='coerce')
            df['date_org_vintage'] = (snapshot_dt - df['date_org']).dt.days / 30.4375

        # 2. Rename Columns based on YAML mapping
        rename_map = self.impute_cfg.get('change_names', {})
        df = df.rename(columns={k.lower(): v for k, v in rename_map.items()})

        # 3. Apply Imputation Strategies
        for item in self.impute_cfg.get('impute_dict', []):
            feat = item['feature'].lower()
            strategy = item['impute_with']
            
            if feat not in df.columns:
                continue

            if strategy == 'CREATE_CATEGORY':
                df[feat] = df[feat].fillna('NULL_CATEGORY')
            
            elif strategy == 'CUST_MEDIAN':
                # Ensure column is numeric before calculating median
                df[feat] = pd.to_numeric(df[feat], errors='coerce')
                median_val = df[feat].median()
                df[feat] = df[feat].fillna(median_val if not pd.isna(median_val) else 0)
            
            elif strategy == 'CUST_MODE':
                mode_res = df[feat].mode()
                df[feat] = df[feat].fillna(mode_res[0] if not mode_res.empty else 'NULL_CATEGORY')

        # Global safety fill for any remaining numeric nulls
        df = df.fillna(0)

        # 4. Manual Dummification (One-Hot Encoding)
        # Based on the categorical_features list in YAML
        cat_features = self.impute_cfg.get('categorical_features', {})
        for feat, values in cat_features.items():
            feat = feat.lower()
            if feat not in df.columns:
                continue
                
            for val in values:
                # Create binary column for each specified value
                col_name = f"{feat}_{str(val).lower()}".replace(" ", "_")
                df[col_name] = (df[feat].astype(str).str.strip() == str(val).strip()).astype(int)
            
            # Optional: Catch-all 'other' category if multiple values are specified
            if len(values) > 1:
                other_name = f"{feat}_other_categories"
                df[other_name] = (~df[feat].astype(str).isin([str(v) for v in values])).astype(int)

        # 5. Clean-up: Drop raw categorical columns and defined 'useless' columns
        useless_cols = [c.lower() for c in self.impute_cfg.get('useless_columns', [])]
        original_cats = [c.lower() for c in cat_features.keys()]
        
        drop_list = list(set(useless_cols + original_cats + ['date_org']))
        df = df.drop(columns=[c for c in drop_list if c in df.columns], errors='ignore')

        self.logger.info(f"Final Matrix Prepared. Shape: {df.shape}")
        return df