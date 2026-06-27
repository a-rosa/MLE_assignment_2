import os
import argparse
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DateType

def process_silver_clickstream(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = os.path.join(bronze_directory, "clickstream", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    _str_cols = {c for c, t in df.dtypes if t == 'string'}
    df = df.select([F.when(col(c) == "", None).otherwise(col(c)).alias(c) if c in _str_cols else col(c) for c in df.columns])
    print('loaded from:', filepath, 'row count:', df.count())

    column_type_map = {"Customer_ID": StringType(), "snapshot_date": DateType()}
    for i in range(1, 21):
        column_type_map[f"fe_{i}"] = IntegerType()
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = os.path.join(silver_directory, "clickstream", partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_clickstream").master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_clickstream(
        args.snapshotdate,
        "datamart/bronze",
        "datamart/silver",
        spark
    )