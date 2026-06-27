import os
from datetime import datetime
import argparse
import pyspark
 
from pyspark.sql.functions import col


def process_bronze_table(snapshot_date_str, csv_file_path, bronze_directory, table_name, spark):
    # create bronze table 
    bronze_table = os.path.join(bronze_directory, table_name)
    if not os.path.exists(bronze_table):
        os.makedirs(bronze_table)

    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str, table_name, 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_table, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

SOURCE_MAP = {
    "clickstream": "data/feature_clickstream.csv",
    "attributes":  "data/features_attributes.csv",
    "financials":  "data/features_financials.csv",
    "lms":         "data/lms_loan_daily.csv",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    parser.add_argument("--source", required=True, choices=SOURCE_MAP.keys())
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName(f"bronze_{args.source}") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_bronze_table(
        args.snapshotdate,
        SOURCE_MAP[args.source],
        "datamart/bronze",
        args.source,
        spark,
    )
    spark.stop()