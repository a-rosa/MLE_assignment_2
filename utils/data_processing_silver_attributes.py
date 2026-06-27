import os
import argparse
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DateType

def strip_chars_and_cast(df, column, new_type, chars="_"):
    """Strip given characters (default: underscore) from a string column, then cast."""
    df = df.withColumn(column, F.regexp_replace(col(column), f"[{chars}]", ""))
    df = df.withColumn(column, col(column).cast(new_type))
    return df

def null_out_values(df, column, bad_values):
    """Replace specific placeholder strings with null."""
    df = df.withColumn(column, F.when(col(column).isin(bad_values), None).otherwise(col(column)))
    return df

def null_out_of_range(df, column, min_val, max_val):
    """Null out values outside a plausible range."""
    df = df.withColumn(
        column,
        F.when((col(column) < min_val) | (col(column) > max_val), None).otherwise(col(column)),
    )
    return df

def process_silver_attributes(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = os.path.join(bronze_directory, "attributes", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    _str_cols = {c for c, t in df.dtypes if t == 'string'}
    df = df.select([F.when(col(c) == "", None).otherwise(col(c)).alias(c) if c in _str_cols else col(c) for c in df.columns])
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.drop("Name", "SSN")

    column_type_map = {
        "Customer_ID": StringType(),
        "Age": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = strip_chars_and_cast(df, "Age", IntegerType())
    df = null_out_of_range(df, "Age", 18, 100)
    df = null_out_values(df, "Occupation", ["_______"])

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    partition_name = "silver_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = os.path.join(silver_directory, "attributes", partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_attributes").master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_attributes(
        args.snapshotdate,
        "datamart/bronze",
        "datamart/silver",
        spark
    )