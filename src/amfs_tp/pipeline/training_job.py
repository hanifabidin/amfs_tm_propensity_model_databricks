import logging
import pickle
import os
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from amfs_tp.data_sources.amfs_handler import AMFSHandler
from amfs_tp.data_sources.bm_handler import BMHandler

class TrainingJob:
    def __init__(self, spark: SparkSession, catalog: str = "amfs_tm", 
                 features_schema: str = "features", models_schema: str = "models"):
        """
        Orchestrates the assembly of training data and model training.
        
        catalog: The Unity Catalog name.
        features_schema: Schema where feature engineering tables are stored.
        models_schema: Schema where training matrices and importance tables are stored.
        """
        self.spark = spark
        self.catalog = catalog
        self.features_schema = f"{catalog}.{features_schema}"
        self.models_schema = f"{catalog}.{models_schema}"
        
        # Managed Volume path for binary model storage
        self.model_volume_path = f"/Volumes/{catalog}/{models_schema}/model_store"
        self.logger = logging.getLogger(__name__)
        
        # Handlers for real-world label retrieval
        self.bm_handler = BMHandler(spark, catalog=catalog)
        self.amfs_handler = AMFSHandler(spark, catalog=catalog)

    def _get_feature_table_names(self):
        """Dynamically identifies all feature tables in the features schema."""
        tables = self.spark.catalog.listTables(self.features_schema)
        return [t.name for t in tables if t.name.endswith("_features")]

    def assemble_matrix(self, snapshot: str, dummy: bool = False):
        """
        Joins labels and all detected features into one wide table.
        
        snapshot: The YYYYMM snapshot of features to use.
        dummy: If True, generates random labels directly from the feature data.
        """
        self.logger.info(f"🚀 Assembling Training Matrix for: {snapshot} (Dummy Mode: {dummy})")
        
        feature_tables = self._get_feature_table_names()
        if not feature_tables:
            raise ValueError(f"No feature tables found in {self.features_schema}")

        # 1. Initialize the Base Matrix (Labels)
        if dummy:
            self.logger.info(f"Generating synthetic labels from {feature_tables[0]}")
            # Use the first feature table as the population base
            training_matrix = (self.spark.table(f"{self.features_schema}.{feature_tables[0]}")
                               .filter(F.col("snapshot_date") == snapshot)
                               .select("cifno", "snapshot_date")
                               .withColumn("label", F.when(F.rand() > 0.9, 1).otherwise(0)))
        else:
            # Standard logic: Get real labels from Call Tracking via Handler
            df_map = self.bm_handler.get_data("cif_acct_mapping", snapshot)
            training_matrix = self.amfs_handler.get_call_labels(snapshot, df_map)

        # 2. Iteratively join all feature tables
        for table_name in feature_tables:
            self.logger.info(f"  Joining feature: {table_name}")
            feat_df = (self.spark.table(f"{self.features_schema}.{table_name}")
                       .filter(F.col("snapshot_date") == snapshot)
                       .drop("snapshot_date"))
            
            # --- PROTECT AGAINST [COLUMN_ALREADY_EXISTS] ---
            # Identify columns that exist in both datasets to avoid ambiguity
            dup_cols = [c for c in feat_df.columns if c in training_matrix.columns and c != "cifno"]
            if dup_cols:
                self.logger.warning(f"    ⚠️ Dropping duplicate columns from {table_name}: {dup_cols}")
                feat_df = feat_df.drop(*dup_cols)
            
            training_matrix = training_matrix.join(feat_df, on="cifno", how="left")

        # 3. Finalize Matrix (Handling nulls from left-joins)
        final_matrix = training_matrix.fillna(0)
        
        # 4. Save Matrix to Models Schema for Audit
        matrix_table_name = f"{self.models_schema}.training_matrix_{snapshot}"
        final_matrix.write.mode("overwrite").saveAsTable(matrix_table_name)
        self.logger.info(f"✅ Training Matrix persisted to: {matrix_table_name}")
        
        return final_matrix

    def train_and_save_model(self, training_matrix, snapshot: str, model_name: str = "propensity_v1.pkl"):
        """
        Trains an XGBoost model using oversampling and saves all artifacts.
        """
        self.logger.info("Preparing data for training with Random Oversampling...")
        
        # 1. Sample to Pandas for small-sample training
        train_pd = training_matrix.toPandas()
        
        # Define target and features
        # Drop non-feature columns
        X = train_pd.drop(['cifno', 'label', 'snapshot_date'], axis=1, errors='ignore')
        y = train_pd['label']
        
        # Ensure only numeric features are used
        X = X.select_dtypes(include=['number'])

        # 2. Handle Class Imbalance
        self.logger.info(f"Pre-sampling distribution: {y.value_counts().to_dict()}")
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        self.logger.info(f"Post-sampling distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

        # 3. Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            random_state=42
        )
        model.fit(X_resampled, y_resampled)

        # 4. Save Feature Importance to Models Schema
        importance_df = pd.DataFrame({
            'feature': X.columns.tolist(),
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        importance_table = f"{self.models_schema}.importance_{snapshot}"
        self.spark.createDataFrame(importance_df).write.mode("overwrite").saveAsTable(importance_table)
        self.logger.info(f"✅ Feature Importance persisted to: {importance_table}")

        # 5. Save Binary Model to Managed Volume
        if not os.path.exists(self.model_volume_path):
            self.logger.info(f"Initializing Volume directory: {self.model_volume_path}")
            # Managed volumes handle root creation, but we ensure path availability
            os.makedirs(self.model_volume_path, exist_ok=True)

        full_pickle_path = os.path.join(self.model_volume_path, model_name)
        
        with open(full_pickle_path, 'wb') as f:
            # We save a dictionary to include the features list for future Inference Jobs
            pickle.dump({
                'model': model,
                'features': X.columns.tolist(),
                'snapshot_trained': snapshot
            }, f)
            
        self.logger.info(f"✅ Pickle model saved to: {full_pickle_path}")
        return full_pickle_path