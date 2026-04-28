import yaml
import logging
import pickle
import os
from pyspark.sql import functions as F
# These are the engine classes we built from your reference files
from amfs_tp.inference.filter_engine import FilterEngine 
from amfs_tp.inference.matrix_engine import MatrixEngine  

class InferenceJob:
    def __init__(self, spark, catalog="amfs_tm", 
                 filter_cfg_path=None, matrix_cfg_path=None, 
                 models_schema="models"):
        """
        catalog: Unity Catalog name 
        filter_cfg_path: Path to your filters_config.yaml
        matrix_cfg_path: Path to your matrix_config.yaml
        """
        self.spark = spark
        self.catalog = catalog
        self.models_schema = f"{catalog}.{models_schema}"
        self.model_volume_path = f"/Volumes/{catalog}/{models_schema}/model_store"
        
        if not filter_cfg_path or not matrix_cfg_path:
            raise ValueError("InferenceJob requires both filter_cfg_path and matrix_cfg_path")
            
        with open(filter_cfg_path, 'r') as f:
            self.filter_cfg = yaml.safe_load(f)
        with open(matrix_cfg_path, 'r') as f:
            self.matrix_cfg = yaml.safe_load(f)

    def run_inference(self, snapshot, model_name="propensity_v1.pkl"):
        self.logger = logging.getLogger(__name__)
        
        # 1. Filtering with Audit Logging (rules.py logic)
        filter_eng = FilterEngine(self.spark, self.catalog, self.filter_cfg)
        filtered_cifs = filter_eng.apply_filters(snapshot)

        # 2. Matrix Generation (merge_all_features.py + impute_dummy.py logic)
        matrix_eng = MatrixEngine(self.spark, self.catalog, self.matrix_cfg)
        final_pd = matrix_eng.create_and_impute(filtered_cifs, snapshot)
        
        # Save dummified matrix for audit
        matrix_table = f"{self.models_schema}.inference_matrix_{snapshot}"
        self.spark.createDataFrame(final_pd).write.mode("overwrite").saveAsTable(matrix_table)

        # 3. Scoring using the Pickle artifacts
        full_model_path = os.path.join(self.model_volume_path, model_name)
        with open(full_model_path, 'rb') as f:
            artifacts = pickle.load(f)
            model = artifacts['model']
            required_features = artifacts['features']

        # Align features: Add missing dummy columns from training
        for col in required_features:
            if col not in final_pd.columns:
                final_pd[col] = 0
        
        X_score = final_pd[required_features]
        final_pd['propensity_score'] = model.predict_proba(X_score)[:, 1]
        
        # 4. Save Final Scored Leads
        output_table = f"{self.models_schema}.scored_leads_{snapshot}"
        leads_df = self.spark.createDataFrame(final_pd[['cifno', 'propensity_score']])
        leads_df.withColumn("snapshot_date", F.lit(snapshot)).write.mode("overwrite").saveAsTable(output_table)
        
        return output_table