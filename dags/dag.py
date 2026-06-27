from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "credit_risk_pipeline",
    default_args=default_args,
    description="Monthly credit risk ML pipeline",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # ── Label store branch ─────────────────────────────────────────
    dep_check_label = DummyOperator(task_id="dep_check_label")

    bronze_lms = BashOperator(
        task_id="bronze_lms",
        bash_command='cd /opt/airflow && python utils/data_processing_bronze_table.py --snapshotdate "{{ ds }}" --source lms',
    )
    silver_loans = BashOperator(
        task_id="silver_loans",
        bash_command='cd /opt/airflow && python utils/data_processing_silver_loans.py --snapshotdate "{{ ds }}"',
    )
    gold_label_store = BashOperator(
        task_id="gold_label_store",
        bash_command='cd /opt/airflow && python utils/data_processing_gold_label_table.py --snapshotdate "{{ ds }}"',
    )
    label_store_completed = DummyOperator(task_id="label_store_completed")

    dep_check_label >> bronze_lms >> silver_loans >> gold_label_store >> label_store_completed

    # ── Feature store branch ───────────────────────────────────────
    dep_check_attributes  = DummyOperator(task_id="dep_check_attributes")
    dep_check_financials  = DummyOperator(task_id="dep_check_financials")
    dep_check_clickstream = DummyOperator(task_id="dep_check_clickstream")

    bronze_attributes = BashOperator(
        task_id="bronze_attributes",
        bash_command='cd /opt/airflow && python utils/data_processing_bronze_table.py --snapshotdate "{{ ds }}" --source attributes',
    )
    bronze_financials = BashOperator(
        task_id="bronze_financials",
        bash_command='cd /opt/airflow && python utils/data_processing_bronze_table.py --snapshotdate "{{ ds }}" --source financials',
    )
    bronze_clickstream = BashOperator(
        task_id="bronze_clickstream",
        bash_command='cd /opt/airflow && python utils/data_processing_bronze_table.py --snapshotdate "{{ ds }}" --source clickstream',
    )

    silver_attributes = BashOperator(
        task_id="silver_attributes",
        bash_command='cd /opt/airflow && python utils/data_processing_silver_attributes.py --snapshotdate "{{ ds }}"',
    )
    silver_financials = BashOperator(
        task_id="silver_financials",
        bash_command='cd /opt/airflow && python utils/data_processing_silver_financials.py --snapshotdate "{{ ds }}"',
    )
    silver_clickstream = BashOperator(
        task_id="silver_clickstream",
        bash_command='cd /opt/airflow && python utils/data_processing_silver_clickstream.py --snapshotdate "{{ ds }}"',
    )

    gold_feature_store = BashOperator(
        task_id="gold_feature_store",
        bash_command='cd /opt/airflow && python utils/data_processing_gold_feature_table.py --snapshotdate "{{ ds }}"',
    )
    feature_store_completed = DummyOperator(task_id="feature_store_completed")

    dep_check_attributes  >> bronze_attributes  >> silver_attributes  >> gold_feature_store
    dep_check_financials  >> bronze_financials  >> silver_financials  >> gold_feature_store
    dep_check_clickstream >> bronze_clickstream >> silver_clickstream >> gold_feature_store
    gold_feature_store >> feature_store_completed

    # ── Model training (gates on both stores) ──────────────────────
    model_automl_start = DummyOperator(task_id="model_automl_start")

    run_model_training = BashOperator(
        task_id="run_model_training",
        bash_command='cd /opt/airflow && python scripts/model_training.py --snapshotdate "{{ ds }}"',
    )
    model_automl_completed = DummyOperator(task_id="model_automl_completed")

    [label_store_completed, feature_store_completed] >> model_automl_start
    model_automl_start >> run_model_training >> model_automl_completed

    # ── Model inference ────────────────────────────────────────────
    model_inference_start = DummyOperator(task_id="model_inference_start")

    run_model_inference = BashOperator(
        task_id="run_model_inference",
        bash_command='cd /opt/airflow && python scripts/model_inference.py --snapshotdate "{{ ds }}"',
    )
    model_inference_completed = DummyOperator(task_id="model_inference_completed")

    model_automl_completed >> model_inference_start >> run_model_inference >> model_inference_completed

    # ── Model monitoring ───────────────────────────────────────────
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    run_model_monitoring = BashOperator(
        task_id="run_model_monitoring",
        bash_command='cd /opt/airflow && python scripts/model_monitoring.py --snapshotdate "{{ ds }}"',
    )
    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")

    model_inference_completed >> model_monitor_start >> run_model_monitoring >> model_monitor_completed