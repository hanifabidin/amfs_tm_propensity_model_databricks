from pyspark.sql.functions import col, when, sum as _sum
from amfs_tp.features.bank.balance import Balance

class BankAvr(Balance):
    def create(self, snapshot):
        df = self.spark.table(f"{self.base_path}.axa_avr_cleaned") \
                 .filter(col("snapshot_date") == snapshot)

        # 1. Rename and fill NA (Pandas median equivalent)
        df = df.withColumnRenamed("nb_accts", "nb_accts_sd").fillna({"sum_end_bal": 0})
        median_val = df.stat.approxQuantile("nb_accts_sd", [0.5], 0.01)[0]
        df = df.fillna({"nb_accts_sd": median_val})

        # 2. Aggregate by CIFNO
        df = df.groupby("cifno").agg(
            _sum("sum_end_bal").alias("sum_end_bal"),
            _sum("nb_accts_sd").alias("nb_accts_sd")
        )

        # 3. Clip accounts at 15
        df = df.withColumn("nb_accts_sd", when(col("nb_accts_sd") > 15, 15).otherwise(col("nb_accts_sd")))

        self._save_to_fs(df, "axa_avr", snapshot)