import argparse
import os
import glob
import pickle
import pprint
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import pyspark
from pyspark.sql.functions import col


def main(snapshotdate):
    print('\n\n---starting model inference job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_inference") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- config ---
    config = {
        "snapshot_date_str": snapshotdate,
        "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d"),
        "model_name": "credit_model_" + snapshotdate.replace("-", "_") + ".pkl",
        "model_bank_directory": "model_bank/",
    }
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    pprint.pprint(config)

    # --- load model artefact ---
    if not os.path.exists(config["model_artefact_filepath"]):
        print(f"No model found at {config['model_artefact_filepath']}. Skipping inference.")
        return
    with open(config["model_artefact_filepath"], "rb") as f:
        model_artefact = pickle.load(f)
    print("model loaded:", config["model_artefact_filepath"])

    # --- load label store for this snapshot date ---
    # Labels use a 6-month outcome window (30dpd_6mob): outcomes for predictions
    # made at snapshot_date are observed at snapshot_date + 6 months.
    LABEL_HORIZON_MONTHS = 6
    label_snapshot_date = config["snapshot_date"] + relativedelta(months=LABEL_HORIZON_MONTHS)
    label_path = "datamart/gold/label_store/"
    label_files = glob.glob(os.path.join(label_path, "*.parquet"))
    label_sdf = spark.read.parquet(*label_files)
    label_sdf = label_sdf.filter(col("snapshot_date") == label_snapshot_date)
    print("labels loaded:", label_sdf.count())

    # --- load feature store filtered to labelled customers only ---
    feature_path = "datamart/gold/feature_store/"
    feature_files = glob.glob(os.path.join(feature_path, "*.parquet"))
    feature_sdf = spark.read.parquet(*feature_files)
    feature_sdf = feature_sdf.filter(col("snapshot_date").cast("date") == config["snapshot_date"].date())
    feature_sdf = feature_sdf.join(label_sdf.select("Customer_ID"), on="Customer_ID", how="inner")
    print("features loaded (labelled customers only):", feature_sdf.count())

    if feature_sdf.count() == 0:
        print(f"No labelled customers found for {snapshotdate} (labels at {label_snapshot_date.date()}). Skipping.")
        spark.stop()
        return

    features_pdf = feature_sdf.toPandas()

    # --- preprocess ---
    feature_cols = model_artefact["feature_cols"]
    X_inference = features_pdf[feature_cols]

    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
    X_inference_p = transformer_stdscaler.transform(X_inference)
    print("X_inference shape:", X_inference_p.shape)

    # --- predict ---
    model = model_artefact["model"]
    y_pred = model.predict_proba(X_inference_p)[:, 1]

    # --- prepare output ---
    output_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    output_pdf["model_name"] = config["model_name"]
    output_pdf["model_predictions"] = y_pred

    # --- save to gold ---
    model_name_no_ext = config["model_name"][:-4]
    gold_directory = f"datamart/gold/model_predictions/{model_name_no_ext}/"
    os.makedirs(gold_directory, exist_ok=True)

    partition_name = (
        model_name_no_ext + "_predictions_" +
        snapshotdate.replace("-", "_") + ".parquet"
    )
    filepath = gold_directory + partition_name
    spark.createDataFrame(output_pdf).write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    spark.stop()
    print('\n\n---completed model inference job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()
    main(args.snapshotdate)