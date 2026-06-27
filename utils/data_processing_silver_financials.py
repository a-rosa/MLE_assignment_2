import os
import argparse
from datetime import datetime

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def strip_chars_and_cast(df, column, new_type, chars="_"):
    """Strip given characters (default: underscore) from a string column, then cast."""
    df = df.withColumn(column, F.regexp_replace(col(column), f"[{chars}]", ""))
    df = df.withColumn(column, F.when(col(column) == "", None).otherwise(col(column)))
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

def parse_credit_history_age(df, column):
    """Convert 'X Years and Y Months' into total months."""
    years = F.regexp_extract(col(column), r"(\d+)\s*Years", 1).cast(IntegerType())
    months = F.regexp_extract(col(column), r"(\d+)\s*Months", 1).cast(IntegerType())
    df = df.withColumn(column, (years * 12 + months).cast(IntegerType()))
    return df

def process_silver_financials(snapshot_date_str, bronze_directory, silver_directory, spark):
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = os.path.join(bronze_directory, "financials", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    _str_cols = {c for c, t in df.dtypes if t == 'string'}
    df = df.select([F.when(col(c) == "", None).otherwise(col(c)).alias(c) if c in _str_cols else col(c) for c in df.columns])
    print('loaded from:', filepath, 'row count:', df.count())

    column_type_map = {
        "Customer_ID": StringType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Total_EMI_per_month": FloatType(),
        "Type_of_Loan": StringType(),
        "Credit_Mix": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Payment_Behaviour": StringType(),
        "snapshot_date": DateType(),
    }
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = strip_chars_and_cast(df, "Annual_Income", FloatType())
    df = strip_chars_and_cast(df, "Num_of_Loan", IntegerType())
    df = strip_chars_and_cast(df, "Num_of_Delayed_Payment", IntegerType())
    df = strip_chars_and_cast(df, "Changed_Credit_Limit", FloatType())
    df = strip_chars_and_cast(df, "Outstanding_Debt", FloatType())
    df = strip_chars_and_cast(df, "Amount_invested_monthly", FloatType())

    df = null_out_values(df, "Credit_Mix", ["_"])
    df = null_out_values(df, "Payment_Behaviour", ["!@9#%8"])
    df = strip_chars_and_cast(df, "Monthly_Balance", FloatType())
    df = null_out_of_range(df, "Num_Bank_Accounts", 0, 999999)
    df = null_out_of_range(df, "Num_of_Loan", 0, 999)
    df = parse_credit_history_age(df, "Credit_History_Age")

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    partition_name = "silver_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = os.path.join(silver_directory, "financials", partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_financials").master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_silver_financials(
        args.snapshotdate,
        "datamart/bronze",
        "datamart/silver",
        spark
    )