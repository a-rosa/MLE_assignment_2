import os
import argparse
import pyspark

from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_attributes import process_silver_attributes
from utils.data_processing_silver_financials import process_silver_financials
from utils.data_processing_silver_clickstream import process_silver_clickstream
from utils.data_processing_silver_loans import process_silver_loans
from utils.data_processing_gold_feature_table import process_gold_feature_store
from utils.data_processing_gold_label_table import process_gold_label_store

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()
    d = args.snapshotdate

    spark = pyspark.sql.SparkSession.builder \
        .appName("pipeline") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Bronze
    process_bronze_table(d, "data/feature_clickstream.csv", "datamart/bronze", "clickstream", spark)
    process_bronze_table(d, "data/features_attributes.csv", "datamart/bronze", "attributes", spark)
    process_bronze_table(d, "data/features_financials.csv", "datamart/bronze", "financials", spark)
    process_bronze_table(d, "data/lms_loan_daily.csv",      "datamart/bronze", "lms", spark)

    # Silver
    process_silver_attributes(d,  "datamart/bronze", "datamart/silver", spark)
    process_silver_financials(d,  "datamart/bronze", "datamart/silver", spark)
    process_silver_clickstream(d, "datamart/bronze", "datamart/silver", spark)
    process_silver_loans(d,       "datamart/bronze", "datamart/silver", spark)

    # Gold
    process_gold_feature_store(d, "datamart/silver", "datamart/gold", spark)
    process_gold_label_store(d,   "datamart/silver", "datamart/gold", spark)