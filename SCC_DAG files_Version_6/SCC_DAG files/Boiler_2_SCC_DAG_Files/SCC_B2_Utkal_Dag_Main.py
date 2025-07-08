from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from DAG_Utkal_SCC_B2_Final import SCCOptimizer

# Setup logging
logger = logging.getLogger(__name__)

# ----------------------- [E] Extract -----------------------
def extract_data(**kwargs):
    model = SCCOptimizer()
    logger.info("Starting data extraction...")
    current_data = model.collect_current_data()
    if current_data is None:
        logger.error("Failed to extract current data")
        raise ValueError("Failed to extract current data")
    
    # Push to XCom
    kwargs['ti'].xcom_push(key='current_data', value=current_data)
    logger.info(f"Data extracted and pushed to XCom: {current_data}")

# ---------------------- [T] Transform ----------------------
def optimize_and_predict(**kwargs):
    model = SCCOptimizer()
    ti = kwargs['ti']
    
    # Pull current_data from XCom
    current_data = ti.xcom_pull(task_ids='extract_data', key='current_data')
    if not current_data:
        logger.error("No current_data found in XCom")
        raise ValueError("Missing current_data in XCom")

    logger.info("Starting optimization process...")
    try:
        x_opt, scc_opt = model.multi_start_optimization(current_data, n_starts=300)
        logger.info("Optimization completed successfully.")
    except Exception as e:
        logger.exception("Optimization failed")
        raise

    scc_model_pred = None
    if model.model:
        try:
            scc_model_pred = model.model.predict([x_opt])[0]
            logger.info(f"Model-predicted SCC: {scc_model_pred:.3f}")
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")

    # Push results to XCom
    ti.xcom_push(key='x_opt', value=x_opt)
    ti.xcom_push(key='scc_opt', value=scc_opt)
    ti.xcom_push(key='scc_model_pred', value=scc_model_pred)
    logger.info("Optimization results pushed to XCom.")

# ------------------------ [L] Load -------------------------
def write_predictions(**kwargs):
    model = SCCOptimizer()
    ti = kwargs['ti']
    
    # Pull values from XCom
    x_opt = ti.xcom_pull(task_ids='optimize_and_predict', key='x_opt')
    scc_opt = ti.xcom_pull(task_ids='optimize_and_predict', key='scc_opt')

    if x_opt is None or scc_opt is None:
        logger.error("Missing optimized values in XCom")
        raise ValueError("Missing optimized values in XCom")

    predictions = {
        "Specific_coal_cons": scc_opt,
        "SCC_PRED": scc_opt,
        "FUEL_AIR_RATIO_PRED": x_opt[0],
        "WIND_BOX_DP_PRED": x_opt[1],
        "PA_FAN_TOT_PRED": x_opt[2],
        "WIND_BOX_AIR_TOT_PRED": x_opt[3],
        "STEAM_FLOW_PRED": x_opt[4],
        "ECO_GAS_TEMP_PRED": x_opt[5],
        "STEAM_TEMP_PRED": x_opt[6],
        "COAL_FLOW_PRED": x_opt[7],
        "ESP_TEMP_PRED": x_opt[8],
        "FDW_FLOW_PRED": x_opt[9],
        "FDW_TEMP_PRED": x_opt[10],
        "O2_PRED": x_opt[11],
        "STM_ENTH_PRED": x_opt[12],
        "FDW_ENTH_PRED": x_opt[13],
    }

    logger.info("Writing predictions to PI...")
    try:
        model.write_predictions_to_pi(predictions)
        logger.info("Predictions successfully written to PI.")
    except Exception as e:
        logger.exception("Failed to write predictions to PI")
        raise

# -------------------- DAG Definition -----------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='utkal_scc_etl_dag',
    default_args=default_args,
    description='ETL DAG for SCC Optimization at Utkal B2',
    schedule_interval='*/10 * * * *',  # every 10 minutes
    catchup=False,
    tags=['utkal', 'scc', 'optimization'],
) as dag:

    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True
    )

    transform_task = PythonOperator(
        task_id='optimize_and_predict',
        python_callable=optimize_and_predict,
        provide_context=True
    )

    load_task = PythonOperator(
        task_id='write_predictions_to_pi',
        python_callable=write_predictions,
        provide_context=True
    )

    extract_task >> transform_task >> load_task
