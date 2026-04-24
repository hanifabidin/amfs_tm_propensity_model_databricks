from pyspark.sql.functions import col, lit
# CRITICAL: This line tells Python where to find the parent class
from amfs_tp.features.bank.balance import Balance

class CreditType(Balance):
    def create_clean(self, snapshot):
        return self.spark.table(f"{self.base_path}.cc_type_cnt_cleaned") \
                   .filter(col("snapshot_date") == snapshot)

    def create(self, snapshot):
        df = self.create_clean(snapshot)
        # Drop partition column and rename columns if needed
        df_final = df.drop("snapshot_date")
        self._save_to_fs(df_final, "credit_type", snapshot)

class DebitType(Balance):
    def create_clean(self, snapshot):
        return self.spark.table(f"{self.base_path}.dc_type_cnt_cleaned") \
                   .filter(col("snapshot_date") == snapshot)

    def create(self, snapshot):
        df = self.create_clean(snapshot)
        df_final = df.drop("snapshot_date")
        self._save_to_fs(df_final, "debit_type", snapshot)