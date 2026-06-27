import os
import glob
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# PSI helper
# ---------------------------------------------------------------------------

def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index between reference and current distributions.
    PSI < 0.1  : stable
    PSI 0.1–0.2: moderate shift (monitor)
    PSI > 0.2  : significant drift (investigate)
    """
    # Build bin edges from reference
    breakpoints = np.linspace(0, 100, bins + 1)
    ref_edges = np.percentile(reference.dropna(), breakpoints)
    ref_edges = np.unique(ref_edges)  # drop duplicate edges
    if len(ref_edges) < 2:
        return 0.0

    # Bin both distributions
    def bin_series(s, edges):
        counts, _ = np.histogram(s.dropna(), bins=edges)
        total = counts.sum()
        if total == 0:
            return np.full(len(counts), 1e-4)
        pct = counts / total
        pct = np.where(pct == 0, 1e-4, pct)  # avoid log(0)
        return pct

    ref_pct = bin_series(reference, ref_edges)
    cur_pct = bin_series(current,   ref_edges)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(snapshotdate: str):
    print("\n\n---starting model monitoring job---\n\n")

    LABEL_HORIZON_MONTHS = 6  # 30dpd_6mob: labels are observed 6 months after prediction
    snapshot_dt = datetime.strptime(snapshotdate, "%Y-%m-%d")
    model_version = "credit_model_" + snapshotdate.replace("-", "_")
    output_dir = os.path.join("monitoring", model_version)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load model artefact (for feature list) ──────────────────
    model_path = os.path.join("model_bank", model_version + ".pkl")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Skipping monitoring.")
        return
    with open(model_path, "rb") as f:
        model_artefact = pickle.load(f)
    feature_cols = model_artefact["feature_cols"]
    print("feature cols:", feature_cols)

    # ── 2. Load all prediction partitions up to snapshotdate ───────
    pred_files = glob.glob(
        f"datamart/gold/model_predictions/{model_version}/*.parquet"
    )
    if not pred_files:
        print(f"No prediction files found for {model_version}. Skipping monitoring.")
        return

    pred_df = pd.read_parquet(pred_files[0])
    for f in pred_files[1:]:
        pred_df = pd.concat([pred_df, pd.read_parquet(f)], ignore_index=True)

    pred_df["snapshot_date"] = pd.to_datetime(pred_df["snapshot_date"]).dt.date
    pred_df = pred_df[pred_df["snapshot_date"] <= snapshot_dt.date()]
    print(f"predictions loaded: {len(pred_df)} rows across {pred_df['snapshot_date'].nunique()} months")

    # ── 3. Load all label partitions up to snapshotdate ────────────
    label_files = glob.glob("datamart/gold/label_store/*.parquet")
    if not label_files:
        raise FileNotFoundError("No label files found")

    label_df = pd.concat([pd.read_parquet(f) for f in label_files], ignore_index=True)
    label_df["snapshot_date"] = pd.to_datetime(label_df["snapshot_date"]).dt.date
    label_cutoff = (snapshot_dt + pd.DateOffset(months=LABEL_HORIZON_MONTHS)).date()
    label_df = label_df[label_df["snapshot_date"] <= label_cutoff]
    print(f"labels loaded: {len(label_df)} rows")

    # ── 4. Load all feature store partitions up to snapshotdate ────
    feat_files = glob.glob("datamart/gold/feature_store/*.parquet")
    if not feat_files:
        raise FileNotFoundError("No feature store files found")

    feat_df = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    feat_df["snapshot_date"] = pd.to_datetime(feat_df["snapshot_date"]).dt.date
    feat_df = feat_df[feat_df["snapshot_date"] <= snapshot_dt.date()]
    print(f"features loaded: {len(feat_df)} rows")

    # ── 5. Gini / AUC over time ────────────────────────────────────
    # Labels use a 6-month outcome window (30dpd_6mob), so the label for a
    # prediction at month T is observed at T+6. Offset pred snapshot_date
    # before joining so we match each prediction to its actual outcome.
    pred_df_eval = pred_df.copy()
    pred_df_eval["label_lookup_date"] = (
        pd.to_datetime(pred_df_eval["snapshot_date"]) + pd.DateOffset(months=LABEL_HORIZON_MONTHS)
    ).dt.date

    eval_df = pred_df_eval.merge(
        label_df[["Customer_ID", "snapshot_date", "label"]].rename(
            columns={"snapshot_date": "label_lookup_date"}
        ),
        on=["Customer_ID", "label_lookup_date"],
        how="inner",
    )
    print(f"eval set (predictions + labels joined): {len(eval_df)} rows")

    monthly_metrics = []
    for dt, grp in eval_df.groupby("snapshot_date"):
        if grp["label"].nunique() < 2:
            continue  # AUC undefined if only one class
        auc = roc_auc_score(grp["label"], grp["model_predictions"])
        gini = round(2 * auc - 1, 4)
        monthly_metrics.append({
            "snapshot_date": dt,
            "auc": round(auc, 4),
            "gini": gini,
            "n": len(grp),
            "bad_rate": round(grp["label"].mean(), 4),
        })

    metrics_df = pd.DataFrame(monthly_metrics)
    if metrics_df.empty:
        print("WARNING: no months with sufficient data for AUC — check prediction/label join")
        return
    metrics_df = metrics_df.sort_values("snapshot_date")
    print(metrics_df.to_string(index=False))

    # Save metrics CSV
    metrics_path = os.path.join(output_dir, "gini_auc_over_time.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"metrics saved to: {metrics_path}")

    # Plot Gini / AUC over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df["snapshot_date"], metrics_df["auc"],  marker="o", label="AUC",  color="steelblue")
    ax.plot(metrics_df["snapshot_date"], metrics_df["gini"], marker="s", label="Gini", color="darkorange")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="AUC=0.5 (random)")
    ax.set_title(f"Model Performance Over Time — {model_version}")
    ax.set_xlabel("Snapshot Date")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    perf_chart_path = os.path.join(output_dir, "gini_auc_over_time.png")
    fig.savefig(perf_chart_path, dpi=150)
    plt.close(fig)
    print(f"performance chart saved to: {perf_chart_path}")

    # ── 6. PSI — feature drift ─────────────────────────────────────
    # Reference = earliest 3 months; current = snapshotdate month
    all_dates = sorted(feat_df["snapshot_date"].unique())
    ref_dates = all_dates[:3]
    ref_df  = feat_df[feat_df["snapshot_date"].isin(ref_dates)]
    cur_df = feat_df[feat_df["snapshot_date"] == snapshot_dt.date()]

    if cur_df.empty:
        print("WARNING: no feature data for snapshotdate — skipping PSI")
    else:
        psi_records = []
        for col in feature_cols:
            if col not in feat_df.columns:
                continue
            psi_val = compute_psi(ref_df[col], cur_df[col])
            flag = (
                "STABLE"   if psi_val < 0.1  else
                "MONITOR"  if psi_val < 0.2  else
                "DRIFT"
            )
            psi_records.append({"feature": col, "psi": psi_val, "flag": flag})

        psi_df = pd.DataFrame(psi_records)
        if psi_df.empty:
            print("WARNING: no months with sufficient data for AUC — check prediction/label join")
            return
        psi_df = pd.DataFrame(psi_records).sort_values("psi", ascending=False)
        print(psi_df.to_string(index=False))

        # Save PSI CSV
        psi_path = os.path.join(output_dir, "psi_feature_drift.csv")
        psi_df.to_csv(psi_path, index=False)
        print(f"PSI saved to: {psi_path}")

        # Plot PSI bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colours = psi_df["flag"].map({"STABLE": "steelblue", "MONITOR": "orange", "DRIFT": "crimson"})
        ax.barh(psi_df["feature"], psi_df["psi"], color=colours)
        ax.axvline(0.1, color="orange", linestyle="--", linewidth=1, label="Monitor threshold (0.1)")
        ax.axvline(0.2, color="crimson", linestyle="--", linewidth=1, label="Drift threshold (0.2)")
        ax.set_title(f"PSI Feature Drift — {model_version}\n(reference: {ref_dates[0]} to {ref_dates[-1]})")
        ax.set_xlabel("PSI")
        ax.legend()
        plt.tight_layout()
        psi_chart_path = os.path.join(output_dir, "psi_feature_drift.png")
        fig.savefig(psi_chart_path, dpi=150)
        plt.close(fig)
        print(f"PSI chart saved to: {psi_chart_path}")

    print("\n\n---completed model monitoring job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", type=str, required=True)
    args = parser.parse_args()
    main(args.snapshotdate)