import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_feature_table(snapshot_date_str, bronze_attributes_directory, bronze_financials_directory, bronze_clickstream_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze tables
    def load_bronze(directory, table_name):
        partition_name = "bronze_" + table_name + "_" + snapshot_date_str.replace('-','_') + '.csv'
        filepath = directory + partition_name
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print('loaded from:', filepath, 'row count:', df.count())
        return df

    df_attributes = load_bronze(bronze_attributes_directory, "attributes")
    df_financials = load_bronze(bronze_financials_directory, "financials")
    df_clickstream = load_bronze(bronze_clickstream_directory, "clickstream")

    # enforce data type for common columns
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    def cast_columns(df):
        for column, new_type in column_type_map.items():
            if column in df.columns:
                df = df.withColumn(column, col(column).cast(new_type))
        return df

    df_attributes = cast_columns(df_attributes)
    df_financials = cast_columns(df_financials)
    df_clickstream = cast_columns(df_clickstream)

    # drop PII columns
    cols_to_drop = [c for c in ["Name", "SSN"] if c in df_attributes.columns]
    df_attributes = df_attributes.drop(*cols_to_drop)

    # join all 3 on Customer_ID and snapshot_date
    df = df_attributes.join(df_financials, on=["Customer_ID", "snapshot_date"], how="left") \
                      .join(df_clickstream, on=["Customer_ID", "snapshot_date"], how="left")

    print('joined row count:', df.count())

    # cast all columns to proper types after join
    column_type_map = {
        "Age": IntegerType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Monthly_Balance": FloatType(),
    }
    
    for c, dtype in column_type_map.items():
        df = df.withColumn(c, col(c).cast(dtype))

    # save silver table
    partition_name = "silver_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df