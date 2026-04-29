import yaml
import logging
import pickle
import os
from pyspark.sql import functions as F
# These are the engine classes built from your reference files
from amfs_tp.inference.filter_engine import FilterEngine 
from amfs_tp.inference.matrix_engine import MatrixEngine

class InferenceJob:
    def __init__(self, spark, catalog="amfs_tm", 
                 filter_cfg_path=None, matrix_cfg_path=None, 
                 impute_cfg_path=None, models_schema="models"):
        """
        Orchestrates the filtering, matrix generation, and scoring pipeline.
        
        catalog: Unity Catalog name 
        filter_cfg_path: Path to filters_config.yaml
        matrix_cfg_path: Path to matrix_config.yaml
        impute_cfg_path: Path to impute_config.yaml
        models_schema: Schema for storing intermediate and final results
        """
        self.spark = spark
        self.catalog = catalog
        self.impute_cfg_path = impute_cfg_path
        self.models_schema = f"{catalog}.{models_schema}"
        self.model_volume_path = f"/Volumes/{catalog}/{models_schema}/model_store"
        
        # Verify all required configuration paths are provided
        if not all([filter_cfg_path, matrix_cfg_path, impute_cfg_path]):
            raise ValueError("InferenceJob requires filter_cfg_path, matrix_cfg_path, and impute_cfg_path")
            
        with open(filter_cfg_path, 'r') as f:
            self.filter_cfg = yaml.safe_load(f)
        with open(matrix_cfg_path, 'r') as f:
            self.matrix_cfg = yaml.safe_load(f)
        

    def run_inference(self, snapshot, model_name="propensity_v1.pkl"):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"--- Starting Inference Pipeline for {snapshot} ---")
        
        # 1. Filtering with Audit Logging (rules.py logic)
        # This step applies business rules and logs CIF counts to the audit schema
        filter_eng = FilterEngine(self.spark, self.catalog, self.filter_cfg)
        filtered_cifs = filter_eng.apply_filters(snapshot)

        # 2. Matrix Generation (merge_all_features.py + impute_dummy.py logic)
        # Now passing the impute_cfg_path to handle YAML-driven imputation/dummification
        matrix_eng = MatrixEngine(self.spark, self.catalog, self.matrix_cfg, self.impute_cfg_path)
        final_pd = matrix_eng.create_and_impute(filtered_cifs, snapshot)
        
        # Save dummified matrix for audit and reproducibility
        matrix_table = f"{self.models_schema}.inference_matrix_{snapshot}"
        self.spark.createDataFrame(final_pd).write.mode("overwrite").saveAsTable(matrix_table)
        self.logger.info(f"Intermediate inference matrix saved to: {matrix_table}")

        # 3. Scoring using the Pickle artifacts
        full_model_path = os.path.join(self.model_volume_path, model_name)
        with open(full_model_path, 'rb') as f:
            artifacts = pickle.load(f)
            model = artifacts['model']
            required_features = artifacts['features']

        # Align features: Ensure all dummy columns from training exist in inference
        # This prevents crashes if specific categories are missing in the current month
        for col in required_features:
            if col not in final_pd.columns:
                final_pd[col] = 0
        
        X_score = final_pd[required_features]
        self.logger.info(f"Generating scores for {len(final_pd)} customers...")
        final_pd['propensity_score'] = model.predict_proba(X_score)[:, 1]
        
        # 4. Save Final Scored Leads
        output_table = f"{self.models_schema}.scored_leads_{snapshot}"
        leads_df = self.spark.createDataFrame(final_pd[['cifno', 'propensity_score']])
        
        # Final output contains CIF and the calculated score, stamped with the snapshot date
        leads_df.withColumn("snapshot_date", F.lit(snapshot)) \
                .write.mode("overwrite") \
                .saveAsTable(output_table)
        
        self.logger.info(f"✅ Inference complete. Final leads saved to: {output_table}")
        
        return output_table