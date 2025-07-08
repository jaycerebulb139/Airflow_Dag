import time
import os
import mlflow
import urllib3
import requests
import warnings
import numpy as np
import pandas as pd
import argparse
from scipy.optimize import minimize
from requests.auth import HTTPBasicAuth
from threading import Lock
from urllib.parse import urlparse
from collections import deque
from minio import Minio
from minio.error import S3Error
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo


# --- Tag Map Loader ---
def load_tag_map_csv_from_minio(minio_client, bucket_name, object_name):
    try:
        response = minio_client.get_object(bucket_name, object_name)
        data = response.read()
        df = pd.read_csv(BytesIO(data))
        return dict(zip(df['tag_name'], df['url']))
    except Exception as e:
        print(f"Failed to load tag map from MinIO: {e}")
        return {}

class SCCOptimizer:
    def __init__(self, bound_strategy="min-max"):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.WINDOW_SEC = 120 # 120 second monitoring cycle
        self.SAMPLE_SEC = 5   # Sample every 5 seconds
        self.ROUND_DIGITS = 3
        self.MAX_SCC = 0.202  # Primary constraint
        self.bound_strategy = bound_strategy

        # Credentials
        self.username = "nilima.gautam-v@adityabirla.com"
        self.password = "Modern@4321"

        # Batch processing variables
        self._batch_urls = set()
        self._batch_cache = {}
        self._batch_lock = Lock()
        self._last_batch_time = 0
        self._batch_interval = 0.5  # seconds

        # MLflow configuration
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
        os.environ["MLFLOW_TRACKING_USERNAME"] = "nilimagautam"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "nilimagautam@2025"
        mlflow.set_tracking_uri("https://hilailab.adityabirla.com/mlflowservice/")

        # MinIO setup
        self.minio_client = Minio(
            endpoint="10.36.42.61:9010",  # Replace with actual endpoint
            access_key="BAmY4n2AG5HyZmuVarrR",         # Replace with actual access key
            secret_key="emMODLUZ4G60ZJp7v4MJVripemfDMHfm2knlg3MQ",         # Replace with actual secret key
            secure=False                           # True if you're using HTTPS
        )
        
        self.bucket_name = "utkal-plant-bucket"
        self.input_tag_map_object = "silver_data_layer/cgpp/Web APIs/SCC_boiler2_read_tag_map.csv"
        self.output_tag_map_object = "silver_data_layer/cgpp/Web APIs/SCC_boiler2_write_tag_map.csv"
        

        # Load input and output tag maps
        self.input_tags = load_tag_map_csv_from_minio(
            self.minio_client,
            self.bucket_name,
            self.input_tag_map_object
        )
        
        self.output_tags = load_tag_map_csv_from_minio(
            self.minio_client,
            self.bucket_name,
            self.output_tag_map_object
        )
        # Load model
        self.model = self.load_model()

        # Model features in exact order
        self.model_features = [
            'HIL_AL_UTKL_CPP_U2_BLR_02_CALC_FUEL_AIR_RATIO', # x[0]
            'HIL_AL_UTKL_CPP_U2_BLR_02_CALC_WIND_BOX_DP',    # x[1]
            'HIL_AL_UTKL_CPP_U2_BLR_02_PA_FAN_PRI_AIR_FL_TOT', # x[2]
            'HIL_AL_UTKL_CPP_U2_BLR_02_FURN_WIND_BOX_AIR_FL_TOT', # x[3]
            'HIL_AL_UTKL_CPP_U2_BLR_02_PRESS_PIPNG_MN_STM_FL_TRNSMTR_01', # x[4]
            'HIL_AL_UTKL_CPP_U2_BLR_02_ECO_FLUE_GAS_TEMP_AVG', # x[5]
            'HIL_AL_UTKL_CPP_U2_BLR_02_SUPR_HTR_MN_STM_HDR_TEMP_02', # x[6]
            'HIL_AL_UTKL_CPP_U2_BLR_02_COAL_FDR_TOT_FLW', # x[7]
            'HIL_AL_UTKL_CPP_U2_BLR_02_ESP_DUCT_INL_FLUE_GAS_TEMP_AVG', # x[8]
            'HIL_AL_UTKL_CPP_U2_BLR_02_FD_WTR_REG_STN_FL_TOT', # x[9]
            'HIL_AL_UTKL_CPP_U2_BLR_02_FD_WTR_REG_STN_TEMP', # x[10]
            'HIL_AL_UTKL_CPP_U2_BLR_02_CALC_OXY', # x[11]
            'HIL_AL_UTKL_CPP_U2_BLR_02_CALC_STM_ENTHLP', # x[12]
            'HIL_AL_UTKL_CPP_U2_BLR_02_CALC_FD_WTR_ENTHLP', # x[13]
        ]

        # Load tags from CSV
        self.input_tags = load_tag_map_csv_from_minio(self.minio_client, self.bucket_name, self.input_tag_map_object)
        self.output_tags = load_tag_map_csv_from_minio(self.minio_client, self.bucket_name, self.output_tag_map_object)
        
        self.input_tag_urls = [self.input_tags[feature] for feature in self.model_features]

        # Rolling buffers for monitoring
        buffer_size = self.WINDOW_SEC // self.SAMPLE_SEC
        self.buffers = {url: deque(maxlen=buffer_size) for url in self.input_tag_urls}

        # Build initial bounds using 1-hour data
        self.bounds = []
        self.build_initial_bounds()

        print(f"Initialized SCC Optimizer with {bound_strategy} strategy")
        print(f"Target: SCC < {self.MAX_SCC}")

    def load_model(self):
        logged_model_uri = "runs:/e8375cecbe1e429e9564c47bfa43c1b4/model"
        return mlflow.xgboost.load_model(logged_model_uri)

    def safe_avg(self, values):
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
        return None

    def _get_batch_url(self, tag_url):
        parsed = urlparse(tag_url)
        return f"{parsed.scheme}://{parsed.netloc}/piwebapi/batch"

    def _batch_fetch(self, urls):
        if not urls:
            return
        batch_url = self._get_batch_url(urls[0])
        batch_payload = {
            f"req{i}": {"Method": "GET", "Resource": url}
            for i, url in enumerate(urls)
        }
        try:
            response = requests.post(
                batch_url,
                json=batch_payload,
                auth=HTTPBasicAuth(self.username, self.password),
                verify=False,
                timeout=10
            )
            response.raise_for_status()
            batch_results = response.json()
            for i, url in enumerate(urls):
                result = batch_results.get(f"req{i}")
                value = None
                if result and "Content" in result:
                    content = result["Content"]
                    if isinstance(content, dict):
                        if "Value" in content:
                            value = content["Value"]
                        elif "Value" in content.get("Value", {}):
                            value = content["Value"]["Value"]
                self._batch_cache[url] = value
        except Exception as e:
            print(f"Batch fetch error: {e}")
            for url in urls:
                self._batch_cache[url] = None

    def read_tag_value(self, url):
        with self._batch_lock:
            now = time.time()
            self._batch_urls.add(url)

            if (now - self._last_batch_time) > self._batch_interval:
                urls_to_fetch = list(self._batch_urls)
                self._batch_urls.clear()
                self._last_batch_time = now
                self._batch_fetch(urls_to_fetch)

            if url in self._batch_cache:
                cached_value = self._batch_cache[url]
                return self._extract_numeric_value(cached_value)

        # fallback to single fetch if not cached
        try:
            response = requests.get(url, auth=HTTPBasicAuth(self.username, self.password), verify=False)
            response.raise_for_status()
            data = response.json()

            value = data.get("Value", None)
            return self._extract_numeric_value(value)

        except Exception as e:
            print(f"Error fetching from {url}: {e}")
            return None

    def _extract_numeric_value(self, value):
        """
        Extract numeric value from the given value field.
        Handles simple numbers or nested dictionaries like:
        {"Name": "...", "Value": 123, ...}
        """
        if isinstance(value, (int, float)):
            return value

        if isinstance(value, dict):
            nested_value = value.get("Value", None)
            if isinstance(nested_value, (int, float)):
                print(f"Extracted nested numeric value: {nested_value}")
                return nested_value
            else:
                print(f"Nested value is not numeric: {nested_value}")
                return None

        print(f"Unexpected value format: {value}")
        return None

    def get_tag_mean_max(self, tag_url, start_time="*-1h", end_time="*"):
        parsed = urlparse(tag_url)
        webid = parsed.path.split("/streams/")[-1].split("/")[0]
        summary_url = f"{parsed.scheme}://{parsed.netloc}/piwebapi/streams/{webid}/summary"
        params = {
            "summaryType": ["Average", "Maximum"],
            "startTime": start_time,
            "endTime": end_time
        }
        try:
            response = requests.get(
                summary_url,
                params=params,
                auth=HTTPBasicAuth(self.username, self.password),
                verify=False,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            mean_val = data['Items'][0]['Value']['Value']
            max_val = data['Items'][1]['Value']['Value']
            if mean_val == max_val:
                delta = abs(mean_val * 0.05) if mean_val != 0 else 1.0
                mean_val -= delta
                max_val += delta
            return round(mean_val, self.ROUND_DIGITS), round(max_val, self.ROUND_DIGITS)
        except Exception as e:
            print(f"Error fetching mean/max for {tag_url}: {e}")
            return None, None

    def build_initial_bounds(self):
        print("Building bounds from 1-hour data for optimization feasibility...")
        self.bounds = []
        for url in self.input_tag_urls:
            mean_val, max_val = self.get_tag_mean_max(url)
            if mean_val is not None and max_val is not None and mean_val != max_val:
                self.bounds.append([mean_val, max_val])
            else:
                self.bounds.append([0, 1])
        print(f"\n=== Optimization Bounds (1-hour data) ===")
        for i, (feature, bound) in enumerate(zip(self.model_features, self.bounds)):
            print(f"x[{i:2d}] {feature:50} [{bound[0]:8.3f}, {bound[1]:8.3f}]")
        coal_bounds = self.bounds[7]
        steam_bounds = self.bounds[4]
        min_scc = coal_bounds[0] / steam_bounds[1]
        max_scc = coal_bounds[1] / steam_bounds[0]
        print(f"\nSCC feasibility: [{min_scc:.6f}, {max_scc:.6f}] vs target < {self.MAX_SCC}")
        if min_scc < self.MAX_SCC:
            print(" SCC < 0.202 is achievable with these bounds")
        else:
            print(" Expanding bounds to make SCC < 0.202 feasible...")
            self.bounds[7][0] = coal_bounds[0] * 0.8
            self.bounds[4][1] = steam_bounds[1] * 1.2

    def scc_objective(self, x):
        return x[7] / x[4]  # coal/steam

    def get_constraints_scc_focused(self, current_data):
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.MAX_SCC - self.scc_objective(x)},
            {'type': 'ineq', 'fun': lambda x: self.scc_objective(x) - 0.200},
            {'type': 'ineq', 'fun': lambda x: x[0] - 4},
            {'type': 'ineq', 'fun': lambda x: x[7] - 35},
            {'type': 'ineq', 'fun': lambda x: x[6] - 120},
            {'type': 'ineq', 'fun': lambda x: x[1] - 10},
            # Delta constraints:
            {'type': 'ineq', 'fun': lambda x: (current_data[13] + 0.101) - x[13]},  
            {'type': 'ineq', 'fun': lambda x: x[13] - (current_data[13] - 0.101)},
            {'type': 'ineq', 'fun': lambda x: (current_data[11] + 0.010) - x[11]},  
            {'type': 'ineq', 'fun': lambda x: x[11] - (current_data[11] - 0.010)},
          #  {'type': 'ineq', 'fun': lambda x: (current_data[6] + 20) - x[6]},   # STEAM TEMP +20°C
          #  {'type': 'ineq', 'fun': lambda x: x[6] - (current_data[6] - 20)},   # STEAM TEMP -20°C
          #  {'type': 'ineq', 'fun': lambda x: (current_data[9] + 50) - x[9]},   # FEED WATER FLOW +50 t/h
          #  {'type': 'ineq', 'fun': lambda x: x[9] - (current_data[9] - 50)},   # FEED WATER FLOW -50 t/h
            # logic to overcome the shooting in delta values         
        ]
        
        return constraints

    def validate_solution(self, x):
        scc_val = self.scc_objective(x)
        if scc_val >= self.MAX_SCC:
            print(f" Solution SCC ({scc_val:.6f}) ≥ limit ({self.MAX_SCC})")
            return False
        return True

    def multi_start_optimization(self, current_data, n_starts=300):
        best_x = None
        best_scc = float('inf')
        successful_runs = 0
        print(f" Running {n_starts} optimization attempts for SCC < {self.MAX_SCC}...")
        for i in range(n_starts):
            x0 = [np.random.uniform(low, high) for low, high in self.bounds]
            if 1:
                res = minimize(
                    self.scc_objective,
                    x0,
                    method='SLSQP',
                    bounds=self.bounds,
                    constraints=self.get_constraints_scc_focused(current_data),
                    #options={'ftol': 1e-6, 'disp': False, 'maxiter': 200}
                )

                if res.success:
                    successful_runs += 1
                    scc_val = self.scc_objective(res.x)
                    # print(scc_val,"scc_val")
                    if scc_val < best_scc and scc_val < self.MAX_SCC:
                        best_x = res.x.copy()
                        best_scc = scc_val
                else:
                    print("Run Not Successful")
            # except Exception:
            #     continue
        if best_x is None:
            raise RuntimeError(f"No feasible solution found with SCC < {self.MAX_SCC}")
        print(f" Optimization complete: {successful_runs}/{n_starts} successful runs")
        print(f" Best SCC achieved: {best_scc:.6f}")
        return np.round(best_x, self.ROUND_DIGITS), round(best_scc, self.ROUND_DIGITS)

    def collect_current_data(self):
        try:
            stored_list = [self.read_tag_value(url) for url in self.input_tags.values()]
            PA_FAN_PRI_AIR_FL_TOT = sum(v for v in stored_list[3:7] if v is not None)
            WIND_BOX_AIR_FL_TOT = sum(v for v in stored_list[8:12] if v is not None)
            ECO_FLUE_GAS_TEMP_AVG = self.safe_avg(stored_list[16:19])
            ESP_DUCT_INL_FLUE_GAS_TEMP_AVG = self.safe_avg(stored_list[21:23])
            FD_WTR_REG_STN_FL_TOT = sum(v for v in stored_list[24:26] if v is not None)
            model_inputs = [
                stored_list[0],   # FUEL_AIR_RATIO
                stored_list[2],   # WIND_BOX_DP
                PA_FAN_PRI_AIR_FL_TOT,
                WIND_BOX_AIR_FL_TOT,
                stored_list[13],  # PRESS_PIPNG_MN_STM_FL_TRNSMTR_01
                ECO_FLUE_GAS_TEMP_AVG,
                stored_list[31],  # SUPR_HTR_MN_STM_HDR_TEMP_02
                stored_list[20],  # COAL_FDR_TOT_FLW
                ESP_DUCT_INL_FLUE_GAS_TEMP_AVG,
                FD_WTR_REG_STN_FL_TOT,
                stored_list[27],  # FD_WTR_REG_STN_TEMP
                stored_list[28],  # CALC_OXY
                stored_list[30],  # CALC_STM_ENTHLP
                stored_list[31],  # CALC_FD_WTR_ENTHLP
            ]
            for i, val in enumerate(model_inputs):
                if val is None or not isinstance(val, (int, float)):
                    print(f" Model input x[{i}] is invalid: {val}")
                    # return None
            return model_inputs
        except Exception as e:
            print(f" Error in data collection: {e}")
            # return None

    def write_predictions_to_pi(self, predictions_dict):
        headers = {"Content-Type": "application/json"}
        for tag_name, value in predictions_dict.items():
            if tag_name in self.output_tags:
                url = self.output_tags[tag_name]

                payload = {"Timestamp": "*", "Value": float(value)}
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        auth=HTTPBasicAuth(self.username, self.password),
                        verify=False,
                        timeout=10
                    )
                    response.raise_for_status()
                    print(f"✓ {tag_name}: {value:.3f}")
                except Exception as e:
                    print(f"✗ Failed to write {tag_name}: {e}")

    def run_optimization_cycle(self):
        ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))
        print(f"\n{'='*80}")
        print(f"SCC OPTIMIZATION CYCLE - {ist_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        current_data = self.collect_current_data()
     
        if current_data is None:
            print(" Skipping cycle: Failed to collect valid data")
            # return
        else:
            print("Current Data=====>",current_data)
            steam_flow = current_data[4]
        
            if steam_flow < 120:
                print(f" Skipping optimization: Steam flow ({steam_flow:.1f}) < 120 t/h")
                # return
            
            current_scc = round(current_data[7] / current_data[4], self.ROUND_DIGITS)
            
            print(f"Current SCC: {current_scc:.3f}")
            
            if current_scc >= self.MAX_SCC:
                print(f" Current SCC ({current_scc:.3f}) ≥ limit ({self.MAX_SCC:.3f}) - Optimization needed!")
            else:
                print(f"✓ Current SCC ({current_scc:.3f}) < limit ({self.MAX_SCC:.3f}) - Within limits")
            try:
                x_opt, scc_opt = self.multi_start_optimization(current_data, n_starts=300)
                
                if self.model:
                    try:
                        scc_model_pred = self.model.predict([x_opt])[0]  # Ensure correct shape
                        print(f"Model-Predicted SCC (on x_opt): {scc_model_pred:.3f}")
                    except Exception as model_error:
                        print(f" Model prediction failed: {model_error}")
                        scc_model_pred = None
                else:
                    scc_model_pred = None
                    
                if not self.validate_solution(x_opt):
                    print(" Optimized solution violates SCC constraint!")
                    return
                print(f"\n=== SCC OPTIMIZATION RESULTS ===")
                print(f"Initial SCC: {current_scc:.3f}")
                print(f"Optimized SCC: {scc_opt:.3f}")
                print(f"Improvement: {current_scc - scc_opt:.3f}")
                print(f"\n=== PARAMETER CHANGES ===")
                param_names = [
                    "Fuel Air Ratio", "Wind Box DP", "PA Fan Total", "Wind Box Air Total",
                    "Steam Flow", "ECO Gas Temp", "Steam Temp", "Coal Flow",
                    "ESP Temp", "Feed Water Flow", "Feed Water Temp", "O2%",
                    "Steam Enthalpy", "Feed Water Enthalpy"
                ]
                
                for i, (name, current, optimized) in enumerate(zip(param_names, current_data, x_opt)):
                    change = optimized - current
                    print(f"x[{i:2d}] {name:20}: {current:8.3f} → {optimized:8.3f} (Δ{change:+7.3f})")
                
                    
                predictions = {
                    # "Specific_coal_cons": scc_model_pred,
                    "SCC_PRED": scc_model_pred,
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
                print(f"\n=== WRITING TO PI ===")
                self.write_predictions_to_pi(predictions)
            except Exception as e:
                print(f" SCC optimization failed: {e}")
    #def run_loop_
    def run_real_time_loop(self):
        print(f" Starting SCC Optimizer")
        print(f"Monitoring: {self.WINDOW_SEC}s window, {self.SAMPLE_SEC}s samples")
        print(f"Target: SCC < {self.MAX_SCC}")
        cycle_start = time.time()
        if 1:
            if 1:
                for url in self.input_tag_urls:
                    value = self.read_tag_value(url)
                    
                    if value is not None:
                        self.buffers[url].append(value)
        
                elapsed = time.time() - cycle_start

                # if elapsed >= self.WINDOW_SEC:
                if 1:
                    self.run_optimization_cycle()
                    cycle_start = time.time()
                    for buffer in self.buffers.values():
                        buffer.clear()
                time.sleep(self.SAMPLE_SEC)

# def main():
#     parser = argparse.ArgumentParser(description="SCC Optimizer - Achieve SCC < 0.202")
#     parser.add_argument("--bounds", choices=["min-max", "mean-max"],
#                         default="min-max",
#                         help="Bounds strategy for monitoring")
#     args = parser.parse_args()
#     optimizer = SCCOptimizer(bound_strategy=args.bounds)
#     optimizer.run_real_time_loop()

# if __name__ == "__main__":
#     main()
