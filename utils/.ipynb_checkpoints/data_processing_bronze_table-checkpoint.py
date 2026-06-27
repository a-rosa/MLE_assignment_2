import os
from datetime import datetime
 
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