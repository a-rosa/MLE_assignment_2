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
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import regexp_extract

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_feature_gold_table(snapshot_date_str, silver_feature_directory, gold_feature_store_directory, spark):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_feature_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # drop column with too many nulls
    df = df.drop("Type_of_Loan")

    # parse credit_history_age column
    df = df.withColumn("Credit_History_Months",
        (regexp_extract(col("Credit_History_Age"), r'(\d+)\s+Year', 1).cast(IntegerType()) * 12) +
        regexp_extract(col("Credit_History_Age"), r'(\d+)\s+Month', 1).cast(IntegerType())
    )
    df = df.drop("Credit_History_Age")

    # replace dirty string values with null
    dirty_map = {
        "Occupation": "______",
        "Credit_Mix": "_",
        "Payment_Behaviour": "!@9#%8"
    }
    for col_name, dirty_val in dirty_map.items():
        df = df.withColumn(col_name, F.when(col(col_name) == dirty_val, None).otherwise(col(col_name)))

    # fix unrealistic numeric values and replace with null
    df = df.withColumn("Age", F.when((col("Age") < 0) | (col("Age") > 100), None).otherwise(col("Age")))

    # replace negatives with null for these columns
    neg_cols = ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", 
                "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment"]
    for c in neg_cols:
        df = df.withColumn(c, F.when(col(c) < 0, None).otherwise(col(c)))

    # fill the nulls
    # numeric columns: fill with median
    numeric_cols = ["Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
                    "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
                    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
                    "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
                    "Amount_invested_monthly", "Monthly_Balance"]
    
    for c in numeric_cols:
        median_val = df.approxQuantile(c, [0.5], 0.01)[0]
        df = df.fillna({c: median_val})

    # categorical columns: fill with mode
    cat_cols = ["Occupation", "Credit_Mix", "Payment_Behaviour", "Payment_of_Min_Amount"]
    for c in cat_cols:
        mode_val = df.groupBy(c).count().orderBy("count", ascending=False).first()[0]
        df = df.fillna({c: mode_val})

    # encode categoricals using StringIndexer
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_encoded", handleInvalid="keep") 
                for c in cat_cols]
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)

    # drop original categorical columns, keep encoded ones
    df = df.drop(*cat_cols)

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df