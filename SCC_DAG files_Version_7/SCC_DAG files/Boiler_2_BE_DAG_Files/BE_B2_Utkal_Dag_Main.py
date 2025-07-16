from datetime import datetime, timedelta
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from DAG_Utkal_BE_B2_Final import SCCOptimizer
import traceback
from scipy.optimize import minimize

# Setup logging
logger = logging.getLogger(__name__)

# ----------------------- [E] Extract -----------------------
def extract_data(**kwargs):
    logger.info("🔹 [Extract] Initializing optimizer and fetching input tag values...")
    try:
        optimizer = SCCOptimizer()
        stored_list = [optimizer.read_tag_value(url) for url in optimizer.input_tags.values()]
        
        if any(v is None for v in stored_list):
            logger.error("[Extract] One or more tag values are missing.")
            raise ValueError("Extraction failed: some tag values are None.")
        
        steam_flow = stored_list[54]
        if steam_flow is None or float(steam_flow) < 120:
            logger.warning("[Extract] Steam flow too low. Skipping pipeline.")
            raise ValueError("Steam flow too low.")
        
        kwargs['ti'].xcom_push(key='stored_list', value=stored_list)
        logger.info("Data extracted and pushed to XCom.")
    except Exception as e:
        logger.error(f"[Extract] Failed: {e}")
        traceback.print_exc()
        raise

# ---------------------- [T] Transform ----------------------
def transform_data(**kwargs):
    logger.info("[Transform] Running prediction and optimization...")
    try:
        ti = kwargs['ti']
        stored_list = ti.xcom_pull(task_ids='extract_data', key='stored_list')
        optimizer = SCCOptimizer()

        Model_inputs = [stored_list[optimizer.var_index.get(feature, 0)] for feature in optimizer.model_features]
        Boiler_eff_prediction = optimizer.model.predict([Model_inputs])[0]
        logger.info(f"Initial Efficiency Prediction: {Boiler_eff_prediction:.3f}")

        optimizer.initialize_bounds()
        result = minimize(optimizer.objective, optimizer.x0, method='SLSQP',
                          bounds=optimizer.bounds, constraints=optimizer.cons)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        x_opt = result.x
        efficiency = -result.fun
        _, loss_dict = optimizer.calculate_losses(x_opt)

        # Your format: short keys, full tag index lookup
        predictions = {
            'MILL_REJ': x_opt[optimizer.var_index['HIL_AL_UTKL_CPP_U2_BLR_02_CALC_MILL_REJCT_LOSS']],
            'BLOW_DOWN': x_opt[optimizer.var_index['HIL_AL_UTKL_CPP_U2_BLR_02_CALC_BLW_DOWN_LOSS']],
            'Dry_Flue_Gas_Loss': loss_dict['L1: Dry Flue Gas Loss'],
            'Radiation_Loss': loss_dict['L10: Radiation Loss'],
            'H2O_in_Air': loss_dict['L4: H2O in Air'],
            'Moisture_in_Fuel': loss_dict['L2: Moisture in Fuel'],
            'H2_in_Fuel': loss_dict['L3: H2 in Fuel'],
            'Total_Unburnt_Loss': loss_dict['UL: Total Unburnt Loss'],
            'Partial_Combustion_Loss': loss_dict['L5: Partial Combustion Loss'],
            'Total_Sensible_Loss': loss_dict['TL: Total Sensible Loss'],
            'Boiler_eff_prediction': efficiency
        }

        logger.info("Optimization completed successfully.")
        ti.xcom_push(key='predictions', value=predictions)

    except Exception as e:
        logger.error(f"[Transform] Failed: {e}")
        traceback.print_exc()
        raise

# ------------------------ [L] Load -------------------------
def load_predictions(**kwargs):
    logger.info("[Load] Writing predictions to PI system...")
    try:
        ti = kwargs['ti']
        predictions = ti.xcom_pull(task_ids='transform_data', key='predictions')
        optimizer = SCCOptimizer()
        optimizer.write_predictions_to_pi(predictions)
        logger.info("Predictions successfully written to PI.")
    except Exception as e:
        logger.error(f"[Load] Failed: {e}")
        traceback.print_exc()
        raise

# -------------------- DAG Definition -----------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025,6,25),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='boiler_efficiency_etl_dag',
    default_args=default_args,
    description='Boiler B3 Efficiency ETL DAG (PI, ML, Optimization)',
    schedule_interval='*/10 * * * *',  # every 10 minutes
    catchup=False,
    tags=['boiler', 'etl', 'efficiency'],
) as dag:

    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True
    )

    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True
    )

    load_task = PythonOperator(
        task_id='load_predictions',
        python_callable=load_predictions,
        provide_context=True
    )

    extract_task >> transform_task >> load_task
