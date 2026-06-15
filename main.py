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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_lms_table
import utils.data_processing_silver_feature_table
import utils.data_processing_gold_lms_table
import utils.data_processing_gold_feature_table

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)

# create bronze lms datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/lms_loan_daily.csv", bronze_lms_directory, "loan_daily", spark)

# create bronze clickstream datalake
bronze_clickstream_directory = "datamart/bronze/clickstream/"

if not os.path.exists(bronze_clickstream_directory):
    os.makedirs(bronze_clickstream_directory)
    
# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/feature_clickstream.csv", bronze_clickstream_directory, "clickstream", spark)

# create bronze attributes datalake
bronze_attributes_directory = "datamart/bronze/attributes/"

if not os.path.exists(bronze_attributes_directory):
    os.makedirs(bronze_attributes_directory)
    
# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/features_attributes.csv", bronze_attributes_directory, "attributes", spark)

# create bronze financials datalake
bronze_financials_directory = "datamart/bronze/financials/"

if not os.path.exists(bronze_financials_directory):
    os.makedirs(bronze_financials_directory)
    
# run bronze backfill
for date_str in dates_str_lst:
    utils.data_processing_bronze_table.process_bronze_table(date_str, "data/features_financials.csv", bronze_financials_directory, "financials", spark)

# create silver lms datalake
silver_loan_daily_directory = "datamart/silver/loan_daily/"

if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)


# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_lms_table.process_silver_lms_table(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# create silver feature datalake
silver_features_directory = "datamart/silver/features/"

if not os.path.exists(silver_features_directory):
    os.makedirs(silver_features_directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_feature_table.process_silver_feature_table(date_str, bronze_attributes_directory, bronze_financials_directory, 
                                                                            bronze_clickstream_directory, silver_features_directory, spark)

# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_lms_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

# create gold datalake
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_feature_table.process_feature_gold_table(date_str, silver_features_directory, gold_feature_store_directory, spark)

