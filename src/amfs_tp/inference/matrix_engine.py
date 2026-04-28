import pandas as pd
from amfs_tp.inference.impute_config import IMPUTE_DICT, CATEGORICAL_FEATURES # Ported from your script

class MatrixEngine:
    def __init__(self, spark, catalog, matrix_config):
        self.spark = spark
        self.catalog = catalog
        self.config = matrix_config

    def create_and_impute(self, population_df, snapshot):
        # 1. Merge Stage (Spark) - From merge_all_features.py
        matrix = population_df
        for name, conf in self.config['feature_sets'].items():
            if not conf.get('enabled'): continue
            
            feat_df = self.spark.table(f"{self.catalog}.{conf['table']}") \
                                .filter(F.col("snapshot_date") == snapshot) \
                                .drop("snapshot_date")
            
            # Collision protection
            dup_cols = [c for c in feat_df.columns if c in matrix.columns and c != "cifno"]
            matrix = matrix.join(feat_df.drop(*dup_cols), "cifno", "left")

        # 2. Impute & Dummy Stage (Pandas) - From impute_dummy.py
        df_pd = matrix.toPandas()
        
        # Apply your specific IMPUTE_DICT logic
        for item in IMPUTE_DICT:
            feat, strategy = item['feature'], item['impute_with']
            if feat in df_pd.columns:
                if strategy == 'CUST_MEDIAN':
                    df_pd[feat] = df_pd[feat].fillna(df_pd[feat].median())
                elif strategy == 'CREATE_CATEGORY':
                    df_pd[feat] = df_pd[feat].fillna('NULL_CATEGORY')

        # Dummification from your CATEGORICAL_FEATURES list
        df_pd = pd.get_dummies(df_pd, columns=list(CATEGORICAL_FEATURES.keys()))
        
        return df_pd