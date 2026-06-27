import argparse
import os
import glob
import pickle
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


# Feature columns from gold feature store
# Attributes: numeric/encoded columns
# Financials: capped numeric columns
FEATURE_COLS = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]


def main(snapshotdate):
    print('\n\n---starting model training job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_training") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- config ---
    model_train_date = datetime.strptime(snapshotdate, "%Y-%m-%d")
    oot_period_months = 3
    train_test_period_months = 9
    train_test_ratio = 0.8

    config = {
        "model_train_date_str": snapshotdate,
        "model_train_date": model_train_date,
        "oot_end_date": model_train_date - timedelta(days=1),
        "oot_start_date": model_train_date - relativedelta(months=oot_period_months),
    }
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    config["train_test_ratio"] = train_test_ratio
    pprint.pprint(config)

    # Labels are stored at observation_date (T+6). Predictions are made at T.
    # Add prediction_date = observation_date - 6 months so we can join with features at T.
    LABEL_HORIZON_MONTHS = 6

    # --- load label store (all partitions) ---
    label_path = "datamart/gold/label_store/"
    label_files = glob.glob(os.path.join(label_path, "*.parquet"))
    label_sdf = spark.read.parquet(*label_files)
    label_sdf = label_sdf.withColumn(
        "prediction_date",
        F.add_months(col("snapshot_date"), -LABEL_HORIZON_MONTHS).cast(DateType())
    )
    label_sdf = label_sdf.filter(
        (col("prediction_date") >= config["train_test_start_date"]) &
        (col("prediction_date") <= config["oot_end_date"])
    )
    print("labels loaded:", label_sdf.count())

    # --- load feature store (all partitions) ---
    feature_path = "datamart/gold/feature_store/"
    feature_files = glob.glob(os.path.join(feature_path, "*.parquet"))
    feature_sdf = spark.read.option("mergeSchema", "true").parquet(*feature_files)
    feature_sdf = feature_sdf.select(["Customer_ID", "snapshot_date"] + FEATURE_COLS)
    feature_sdf = feature_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    feature_sdf = feature_sdf.withColumnRenamed("snapshot_date", "prediction_date")
    print("features loaded:", feature_sdf.count())

    # --- join features + labels on Customer_ID + prediction_date ---
    data_pdf = label_sdf.join(feature_sdf, on=["Customer_ID", "prediction_date"], how="left").toPandas()

    # resolve feature cols that actually exist
    available_features = [c for c in FEATURE_COLS if c in data_pdf.columns]
    print("feature cols used:", len(available_features))

    # --- split: train/test vs OOT ---
    oot_mask = (
        (data_pdf["prediction_date"] >= config["oot_start_date"].date()) &
        (data_pdf["prediction_date"] <= config["oot_end_date"].date())
    )
    oot_pdf = data_pdf[oot_mask]
    train_test_pdf = data_pdf[~oot_mask]

    if len(train_test_pdf) == 0 or len(oot_pdf) == 0:
        print(f"Insufficient training data at {snapshotdate} "
              f"(train_test={len(train_test_pdf)}, oot={len(oot_pdf)}). Skipping.")
        spark.stop()
        return

    X_oot = oot_pdf[available_features]
    y_oot = oot_pdf["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[available_features], train_test_pdf["label"],
        test_size=1 - train_test_ratio,
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )

    print(f"X_train: {X_train.shape[0]}, X_test: {X_test.shape[0]}, X_oot: {X_oot.shape[0]}")

    # --- preprocessing ---
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train)
    X_train_p = transformer_stdscaler.transform(X_train)
    X_test_p  = transformer_stdscaler.transform(X_test)
    X_oot_p   = transformer_stdscaler.transform(X_oot)

    # --- train XGBoost with RandomizedSearchCV ---
    xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=88)

    param_dist = {
        "n_estimators": [25, 50],
        "max_depth": [2, 3],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "gamma": [0, 0.1],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 1.5, 2],
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=make_scorer(roc_auc_score),
        n_iter=20,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=2,
    )
    random_search.fit(X_train_p, y_train)
    best_model = random_search.best_estimator_

    # --- evaluate ---
    def gini(y_true, y_prob):
        return round(2 * roc_auc_score(y_true, y_prob) - 1, 3)

    train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train_p)[:, 1])
    test_auc  = roc_auc_score(y_test,  best_model.predict_proba(X_test_p)[:, 1])
    oot_auc   = roc_auc_score(y_oot,   best_model.predict_proba(X_oot_p)[:, 1])

    print(f"Train AUC: {train_auc:.4f} | Gini: {gini(y_train, best_model.predict_proba(X_train_p)[:,1])}")
    print(f"Test  AUC: {test_auc:.4f}  | Gini: {gini(y_test,  best_model.predict_proba(X_test_p)[:,1])}")
    print(f"OOT   AUC: {oot_auc:.4f}   | Gini: {gini(y_oot,   best_model.predict_proba(X_oot_p)[:,1])}")

    # --- save artefact ---
    model_artefact = {
        "model": best_model,
        "model_version": "credit_model_" + snapshotdate.replace("-", "_"),
        "feature_cols": available_features,
        "preprocessing_transformers": {"stdscaler": transformer_stdscaler},
        "data_dates": {k: str(v) for k, v in config.items()},
        "data_stats": {
            "X_train": X_train.shape[0], "X_test": X_test.shape[0], "X_oot": X_oot.shape[0],
            "y_train_mean": round(y_train.mean(), 3),
            "y_test_mean":  round(y_test.mean(), 3),
            "y_oot_mean":   round(y_oot.mean(), 3),
        },
        "results": {
            "auc_train": train_auc, "auc_test": test_auc, "auc_oot": oot_auc,
            "gini_train": gini(y_train, best_model.predict_proba(X_train_p)[:,1]),
            "gini_test":  gini(y_test,  best_model.predict_proba(X_test_p)[:,1]),
            "gini_oot":   gini(y_oot,   best_model.predict_proba(X_oot_p)[:,1]),
        },
        "hp_params": random_search.best_params_,
    }
    pprint.pprint(model_artefact)

    os.makedirs("model_bank", exist_ok=True)
    file_path = os.path.join("model_bank", model_artefact["model_version"] + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(model_artefact, f)
    print(f"Model saved to {file_path}")

    spark.stop()
    print('\n\n---completed model training job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()
    main(args.snapshotdate)