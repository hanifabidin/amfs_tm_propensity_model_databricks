import yaml
import importlib
import logging
from pyspark.sql import SparkSession

class FeatureJob:
    def __init__(self, spark: SparkSession, catalog="amfs_tm", clean_schema="clean", features_schema="features", config_path=None):
        """
        Orchestrates the Feature Engineering pipeline.
        
        catalog: The catalog name (e.g., amfs_tm)
        clean_schema: Source schema containing validated BM/AMFS data (and legacy views)
        features_schema: Target schema for registered feature tables
        config_path: Path to feature_config.yaml
        """
        self.spark = spark
        self.catalog = catalog
        self.clean_schema = clean_schema
        self.features_schema = features_schema
        self.logger = logging.getLogger(__name__)
        
        if not config_path:
            raise ValueError("FeatureJob requires a config_path for feature_config.yaml")
            
        with open(config_path, 'r') as f:
            self.feat_cfg = yaml.safe_load(f)

    def run(self, snapshot):
        """
        Executes the modules and methods defined in feature_config.yaml sequentially.
        """
        self.logger.info(f"--- Starting Feature Engineering Pipeline for {snapshot} ---")
        
        for item in self.feat_cfg['feature_modules']:
            # Respect the 'enabled' flag from YAML
            if not item.get('enabled', False):
                self.logger.info(f"Skipping disabled module: {item.get('class', 'Unknown')}")
                continue

            module_path = item['module']
            class_name = item['class']
            methods_to_run = item['methods']

            try:
                # 1. Dynamic Import
                # Loads the module (e.g., amfs_tp.features.bank.phsumm)
                module = importlib.import_module(module_path)
                FeatureClass = getattr(module, class_name)
                
                # 2. Instantiate with 'clean' as the input schema
                # This ensures the class finds input tables in the clean schema via base_path
                feature_instance = FeatureClass(self.spark, self.catalog, self.clean_schema)
                
                # 3. SCHEMA REDIRECT (Dual Injection)
                
                # A. Inject into the instance itself (for classes using _save_to_fs from balance.py)
                # Since classes like BankPH inherit from Balance, this overrides the output path
                feature_instance.target_schema = self.features_schema
                
                # B. Inject into the manager (for classes using register_or_update)
                if hasattr(feature_instance, 'manager'):
                    feature_instance.manager.target_schema = self.features_schema
                
                # 4. Execute sequential methods defined in YAML
                for method_name in methods_to_run:
                    if hasattr(feature_instance, method_name):
                        self.logger.info(f"  Executing {class_name}.{method_name} (Snapshot: {snapshot})")
                        getattr(feature_instance, method_name)(snapshot)
                    else:
                        self.logger.warning(f"  Method {method_name} not found in class {class_name}")
                        
            except Exception as e:
                self.logger.error(f"❌ Critical error in module {class_name}: {str(e)}")
                # Fail the entire job if a feature module crashes to prevent data inconsistency
                raise e
        
        self.logger.info(f"--- Feature Engineering Pipeline Completed Successfully for {snapshot} ---")