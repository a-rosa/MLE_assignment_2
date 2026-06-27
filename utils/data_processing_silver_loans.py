import os
import argparse
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_loans(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = "bronze_lms_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = os.path.join(bronze_directory, "lms", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    _str_cols = {c for c, t in df.dtypes if t == 'string'}
    df = df.select([F.when(col(c) == "", None).otherwise(col(c)).alias(c) if c in _str_cols else col(c) for c in df.columns])
    print('loaded from:', filepath, 'row count:', df.count())

    column_type_map = {
        "loan_id": StringType(), "Customer_ID": StringType(),
        "loan_start_date": DateType(), "tenure": IntegerType(),
        "installment_num": IntegerType(), "loan_amt": FloatType(),
        "due_amt": FloatType(), "paid_amt": FloatType(),
        "overdue_amt": FloatType(), "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "installments_missed",
        F.when(col("due_amt") > 0, F.ceil(col("overdue_amt") / col("due_amt"))).otherwise(0).cast(IntegerType())
    ).fillna(0)
    df = df.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0,
               F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType())
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date")))
         .otherwise(0).cast(IntegerType())
    )

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    partition_name = "silver_lms_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = os.path.join(silver_directory, "loans", partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_loans").master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_loans(
        args.snapshotdate,
        "datamart/bronze",
        "datamart/silver",
        spark
    )