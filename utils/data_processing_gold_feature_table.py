import os
import glob
from datetime import datetime
import argparse
import pyspark

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# ---------------------------------------------------------------------------
# Outlier caps — computed once from full silver history (2023-01 to 2024-12)
# Hard-coded here so thresholds never drift between partitions (avoids PSI noise
# and train-serve skew)
# ---------------------------------------------------------------------------
OUTLIER_CAPS = {
    "Num_Credit_Card":         {"lower": 1,      "upper": 837},
    "Interest_Rate":           {"lower": 1,      "upper": 2660},
    "Num_Credit_Inquiries":    {"lower": 0,      "upper": 762},
    "Num_of_Loan":             {"lower": 0,      "upper": 9},
    "Num_of_Delayed_Payment":  {"lower": 0,      "upper": 27},
    "Annual_Income":           {"lower": 7518,   "upper": 177908},
    "Outstanding_Debt":        {"lower": 25,     "upper": 4793},
    "Amount_invested_monthly": {"lower": 17,     "upper": 10000},
    "Monthly_Inhand_Salary":   {"lower": 529,    "upper": 13712},
    "Total_EMI_per_month":     {"lower": 0,      "upper": 56524},
    "Num_Bank_Accounts":       {"lower": 0,      "upper": 330},
}


def cap_outliers(df, caps):
    """Winsorise columns to hard-coded [lower, upper] bounds."""
    for column, bounds in caps.items():
        if column in df.columns:
            df = df.withColumn(
                column,
                F.when(col(column) < bounds["lower"], bounds["lower"])
                 .when(col(column) > bounds["upper"], bounds["upper"])
                 .otherwise(col(column))
            )
    return df


def impute_nulls(df, spark):
    """
    Median imputation for numerics, mode for categoricals.
    Nulls from outlier-capping and dirty strings are resolved here (Gold
    responsibility, not Silver).
    """
    for c in df.columns:
        dtype = dict(df.dtypes)[c]

        if dtype in ("int", "bigint", "float", "double"):
            result = df.approxQuantile(c, [0.5], 0.001)
            if not result:  # all-null column, skip
                continue
            median_val = result[0]
            df = df.withColumn(c, F.when(col(c).isNull(), median_val).otherwise(col(c)))

        elif dtype == "string":
            mode_row = (
                df.groupBy(c).count()
                  .orderBy(F.desc("count"))
                  .filter(col(c).isNotNull())
                  .first()
            )
            if mode_row:
                df = df.withColumn(c, F.when(col(c).isNull(), mode_row[0]).otherwise(col(c)))

    return df


def process_gold_feature_store(snapshot_date_str, silver_directory, gold_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # --- load silver partitions ---
    def load_silver(source):
        partition_name = f"silver_{source}_{snapshot_date_str.replace('-', '_')}.parquet"
        filepath = os.path.join(silver_directory, source, partition_name)
        df = spark.read.parquet(filepath)
        print(f"loaded {source} from: {filepath}, row count: {df.count()}")
        return df

    df_attr  = load_silver("attributes")
    df_fin   = load_silver("financials")
    df_click = load_silver("clickstream")

    # --- cap outliers on financials before join ---
    df_fin = cap_outliers(df_fin, OUTLIER_CAPS)

    # --- drop columns not used downstream ---
    # Type_of_Loan: 20+ multi-value categorical, encoding impractical
    df_fin = df_fin.drop("Type_of_Loan")

    # --- join: attributes -> financials -> clickstream ---
    # all left joins keyed on Customer_ID; clickstream covers all customers
    # so ~530 active loan rows per snapshot is expected
    df = df_attr.join(df_fin,   on="Customer_ID", how="left") \
                .join(df_click, on="Customer_ID", how="left")

    # --- drop duplicate snapshot_date columns introduced by join ---
    # keep the first, drop the rest
    snapshot_cols = [c for c in df.columns if c == "snapshot_date"]
    for dup in snapshot_cols[1:]:
        df = df.drop(dup)
    
    # --- impute nulls ---
    df = impute_nulls(df, spark)

    # --- add snapshot_date column ---
    df = df.withColumn("snapshot_date", F.lit(snapshot_date_str).cast(DateType()))

    # --- cast fe_ columns to double for schema consistency across partitions ---
    fe_cols = [c for c in df.columns if c.startswith("fe_")]
    for c in fe_cols:
        df = df.withColumn(c, col(c).cast(FloatType()))

    # --- save ---
    gold_feature_path = os.path.join(gold_directory, "feature_store")
    if not os.path.exists(gold_feature_path):
        os.makedirs(gold_feature_path)

    partition_name = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_feature_path, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f"saved to: {filepath}, row count: {df.count()}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_feature_store") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process_gold_feature_store(
        args.snapshotdate,
        "datamart/silver",
        "datamart/gold",
        spark,
    )
    spark.stop()