import os
from datetime import datetime
import argparse
import pyspark

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType

# Label definition constants
DPD = 30
MOB = 6

def process_gold_label_store(snapshot_date_str, silver_directory, gold_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load silver loans partition
    partition_name = f"silver_lms_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_directory, "loans", partition_name)
    df = spark.read.parquet(filepath)
    print(f"loaded from: {filepath}, row count: {df.count()}")

    # filter to MOB=6 rows only — these are customers 6 months into their loan
    # snapshot_date on this row is already the label timestamp, no date arithmetic needed
    df = df.filter(col("mob") == MOB)

    # label: 1 if DPD >= 30 at MOB=6, else 0
    df = df.withColumn(
        "label",
        F.when(col("dpd") >= DPD, 1).otherwise(0).cast(IntegerType())
    )
    df = df.withColumn(
        "label_def",
        F.lit(f"{DPD}dpd_{MOB}mob").cast(StringType())
    )

    # keep only what downstream needs
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save
    gold_label_path = os.path.join(gold_directory, "label_store")
    if not os.path.exists(gold_label_path):
        os.makedirs(gold_label_path)

    partition_name = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_label_path, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f"saved to: {filepath}, row count: {df.count()}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_label_store") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_gold_label_store(
        args.snapshotdate,
        "datamart/silver",
        "datamart/gold",
        spark,
    )
    spark.stop()