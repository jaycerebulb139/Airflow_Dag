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
        # self.MAX_SCC = 0.202  # Primary constraint
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
        self.input_tag_map_object = "silver_data_layer/cgpp/Web APIs/BE_Boiler_1_read_tags.csv"
        self.output_tag_map_object = "silver_data_layer/cgpp/Web APIs/BE_Boiler_1_write_tags.csv"
        

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

 
        # Model features in exact order    
        self.model_features = [
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_TOT_LOSS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_HEAT_LOSS_IN_DRY_FLUE_GAS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_TOT',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_SPECEFIC_COAL_CONS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_NOX',
            'HIL_AL_UTKL_CPP_U1_BLR_01_ESP_OUTL_GAS_TEMP_AVG',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_AIR',
            'HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_AVG',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_FUEL',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_CO',
            'HIL_AL_UTKL_CPP_U1_BLR_01_SUPR_HTR_MN_STM_HDR_TEMP_02',
            'HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_FAN_TOT'
        ]

        self.var_to_tag_mapping = {
            "Tfg": "Tfg", "Tair": "FDfanInlet", "H2": "H2", "Excess_Air":"Excess_Air","Theoritical_Air":"Theoritical_Air",
            "CO": "CO_APH_OUT", "ASH": "ASH", "TM": "TM", "GCV": "GCV", "C":"C","feedwaterfl":"feedwaterfl",
            "SteamGen":"SteamGen","RejectQuality":"RejectQuality", "RejectGCV":"RejectGCV","SteamEnthalpy":"SteamEnthalpy",
            "FeedwEnthalpy":"FeedwEnthalpy","S":"S","N2":"N2",'Radiation_Loss':'Radiation_Loss','BottomAsh':'BottomAsh',
            'EspAsh':'EspAsh','Blow_down_loss':'Blow_down_loss',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_TOT_LOSS': 'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_TOT_LOSS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS': 'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS':'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS' ,
            'HIL_AL_UTKL_CPP_U1_BLR_01_HEAT_LOSS_IN_DRY_FLUE_GAS': 'HIL_AL_UTKL_CPP_U1_BLR_01_HEAT_LOSS_IN_DRY_FLUE_GAS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS': 'HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_TOT': 'HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_TOT',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_SPECEFIC_COAL_CONS': 'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_SPECEFIC_COAL_CONS',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_NOX': 'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_NOX',
            'HIL_AL_UTKL_CPP_U1_BLR_01_ESP_OUTL_GAS_TEMP_AVG': 'HIL_AL_UTKL_CPP_U1_BLR_01_ESP_OUTL_GAS_TEMP_AVG',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_AIR': 'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_AIR',
            'HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_AVG': 'HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_AVG',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_FUEL': 'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_FUEL',
            'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_CO':'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_CO',
            'HIL_AL_UTKL_CPP_U1_BLR_01_SUPR_HTR_MN_STM_HDR_TEMP_02': 'HIL_AL_UTKL_CPP_U1_BLR_01_SUPR_HTR_MN_STM_HDR_TEMP_02',
            'HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_FAN_TOT': 'HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_FAN_TOT',
            'Main_Steam_flow_trans_01':'Main_Steam_flow_trans_01'
        }

        # Step 4: Indexing
        self.var_index = {
            var: idx for idx, var in enumerate(self.var_to_tag_mapping)
        }
        print("var_index--", self.var_index)
        self.var_to_tag_url = {
            var: self.input_tags.get(tag_name, "")  # Might be "" if not found
            for var, tag_name in self.var_to_tag_mapping.items()
        }
        # self.C = 40.49  # Constant C for L5 formula

        self.cons = [
        {'type': 'ineq', 'fun': lambda x: x[self.var_index['Tfg']] - x[self.var_index['Tair']] - 10},
        # {'type': 'ineq', 'fun': lambda x: (x[self.var_index['AAS']] / x[self.var_index['Mdfg']]) - 1.5},
        {'type': 'ineq', 'fun': lambda x: x[self.var_index['TM']] - x[self.var_index['H2']]}
        ]
        # Load tags from CSV
        self.input_tags = load_tag_map_csv_from_minio(self.minio_client, self.bucket_name, self.input_tag_map_object)
        self.output_tags = load_tag_map_csv_from_minio(self.minio_client, self.bucket_name, self.output_tag_map_object)

        # self.x0 = np.array([np.mean(b) for b in self.bounds])
        
        self.input_tag_urls = [self.input_tags[feature] for feature in self.model_features]

        # Rolling buffers for monitoring
        # buffer_size = self.WINDOW_SEC // self.SAMPLE_SEC
        # self.buffers = {url: deque(maxlen=buffer_size) for url in self.input_tag_urls}

        # Build initial bounds using 1-hour data
        # self.bounds = []
        # self.build_initial_bounds()

        # This must come first
        self.initialize_bounds()
        
        # Now that bounds are initialized, we can safely access them
        # self.x0 = np.array([np.mean(b) for b in self.bounds])
        self.x0 = [np.random.uniform(low, high) for low, high in self.bounds]
        
        self.model = self.load_model()
        
        print(f"Initialized SCC Optimizer with {bound_strategy} strategy")
        # print(f"Target: SCC < {self.MAX_SCC}")

    def load_model(self):
        logged_model_uri = "runs:/44a732877191499d86d4d6a5595be164/model"
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
    
    def get_dynamic_bounds_by_index(self, index, buffer=0.1):
        try:
            variable = list(self.var_index.keys())[index]
            # print("variable---", variable)
            tag_url = self.var_to_tag_url.get(variable)
    
            if not tag_url:
                print(f"[Missing URL] No input tag URL for variable '{variable}'")
                return (0.0, 1.0)
    
            mean_val, max_val = self.get_tag_mean_max(tag_url)
            if mean_val is None or max_val is None:
                return (0.0, 1.0)
    
            # Ensure valid ordering
            low = min(mean_val, max_val)
            high = max(mean_val, max_val)
    
            delta = abs(high - low) * buffer
            min_bound = max(0.0, low - delta)
            max_bound = high + delta
    
            # Final validation
            if max_bound < min_bound:
                print(f"[Bound Warning] Adjusted invalid bounds for index {index}: {min_bound} > {max_bound}")
                return (0.0, 1.0)
    
            return (round(min_bound, self.ROUND_DIGITS), round(max_bound, self.ROUND_DIGITS))
    
        except Exception as e:
            print(f"[Bound Error] Index {index}: {e}")
            return (0.0, 1.0)
    
                
    def initialize_bounds(self):
        self.bounds = []
        for i in range(len(self.var_index)):
            bound = self.get_dynamic_bounds_by_index(i)
            self.bounds.append(bound)
                
    
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
        
    def calculate_losses(self, x):
        # Unpack tunable variables from x
        Tfg, Tair,H2, Excess_Air, Theoritical_Air,CO, ASH, TM, GCV,C, feedwaterfl, SteamGen, RejectQuality, RejectGCV, SteamEnthalpy, FeedwEnthalpy,S, N2,Radiation_Loss,BottomAsh,EspAsh,Blow_down_loss= x[:22]


        # Fetch from API using your tag mapping and reader
        def get_var(tag_key):
            url = self.var_to_tag_url.get(tag_key)
            
            if not url:
                print(f"[Missing URL] No URL found for variable: {tag_key}")
                # return None
            val = self.read_tag_value(url)
            if val is None:
                print(f"[Missing Value] No data for tag: {tag_key}")
            return val
    
        # Read real-time variables
        C = get_var("C")
        N2 = get_var("N2")
        H2 = get_var("H2")
        S = get_var("S")
    
        # Fail-safe: if any are None, abort calculation
        if None in [C, N2, H2, S]:
            print("[Abort] Missing critical inputs from API.")
            return float("inf"), {}
    
        # Proceed with calculations
        AF = 0.016
        AAS = (1 + Excess_Air / 100) * Theoritical_Air
        Mdfg = ((C * 44 / 12) + (S * 64 / 32)+N2 +(AAS*77)+ ((AAS - Theoritical_Air) * 23)) / 100
        L1 = ((Mdfg * 0.24) * (Tfg - Tair) / GCV) * 100
        L2 = (TM * (584 + 0.45 * (Tfg - Tair))) / GCV
        L3 = 8.937 * (584 + 0.45 * (Tfg - Tair)) * H2 / GCV
        L4 = (AF * AAS * 0.45 * (Tfg - Tair) * 100) / GCV
        L5 = 0.02
        L6 = 0.85*ASH*EspAsh*8077/(100*GCV)
        L7 = ((ASH * 15 / 100) * BottomAsh * 8077) / (100 * GCV)
        L8 = ((ASH * 80 / 10000) * 0.23 * (Tfg - Tair) * 100) / GCV
        L9 = ((ASH * 20 / 10000) * 0.23 * (Tfg - Tair) * 100) / GCV
        L10 = Radiation_Loss
        # L11 = (((0.8 * (feedwaterfl - SteamGen) * 300) + (0.2 * (feedwaterfl - SteamGen) * 150)) * 100) / ((SteamGen * (SteamEnthalpy - FeedwEnthalpy)))
        L11 = Blow_down_loss
        L12 = (RejectQuality * RejectGCV * 100) / ((SteamGen * (SteamEnthalpy - FeedwEnthalpy)) * 1000 / 0.85)
    
        UL = L6 + L7
        SL = L8 + L9
        TL = SL
        total = L1 + L2 + L3 + L4 + L5 + UL + SL + L10
    
        loss_dict = {
            'Total Loss': total,
            'L1: Dry Flue Gas Loss': L1,
            'L2: Moisture in Fuel': L2,
            'L3: H2 in Fuel': L3,
            'L4: H2O in Air': L4,
            'L5: Partial Combustion Loss': L5,
            'L6: Unburnt ESP Loss': L6,
            'L7: Unburnt bottom ash loss': L7,
            'UL: Total Unburnt Loss': UL,
            'L8: Sensible ESP loss': L8,
            'L9: Sensible Bottom loss': L9,
            'TL: Total Sensible Loss': TL,
            'L10: Radiation Loss': L10,
            'L11: BlowDown Loss': L11,
            'L12: Mill Rejection Loss': L12
        }


        return total, loss_dict

    def objective(self, x):
        total_loss, loss_dict = self.calculate_losses(x)

        feature_map = {
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_TOT_LOSS': total_loss,
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS': x[12],
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS': x[13],
            'HIL_AL_UTKL_CPP_U1_BLR_01_HEAT_LOSS_IN_DRY_FLUE_GAS': loss_dict['L1: Dry Flue Gas Loss'],
            'HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS': x[14],
            'HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_TOT': x[15],
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_SPECEFIC_COAL_CONS': x[16],
            'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_NOX': x[17],
            'HIL_AL_UTKL_CPP_U1_BLR_01_ESP_OUTL_GAS_TEMP_AVG': x[18],
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_AIR': loss_dict['L4: H2O in Air'],
            'HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_AVG': x[19],
            'HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_FUEL': loss_dict['L2: Moisture in Fuel'],
            'HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_CO': x[5],
            'HIL_AL_UTKL_CPP_U1_BLR_01_SUPR_HTR_MN_STM_HDR_TEMP_02': x[20],
            'HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_FAN_TOT': x[21]
        }

        input_df = pd.DataFrame([feature_map])[self.model_features]

        efficiency = self.model.predict(input_df)[0]
        return -efficiency  # Minimize for maximization


    def run_loop(self):
        if 1:
            stored_list = [self.read_tag_value(url) for url in self.input_tags.values()]  ### Fetching realtime data

            print("")
            for i, tag in enumerate(self.input_tags.keys()):
                print(f"{tag}: {stored_list[i]}")

            # This sequence for fetching data from server
            # Assign variables (adjust length as per your tags)
            HIL_AL_UTKL_CPP_U1_BLR_01_CALC_TOT_LOSS = stored_list[0]
            HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS = stored_list[1]
            HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS = stored_list[2]
            HIL_AL_UTKL_CPP_U1_BLR_01_HEAT_LOSS_IN_DRY_FLUE_GAS = stored_list[3]
            HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS = stored_list[4]
            HIL_AL_UTKL_CPP_U1_BLR_01_CALC_SPECEFIC_COAL_CONS = stored_list[5]
            HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_NOX = stored_list[6]
            HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_AIR = stored_list[7]
            HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_FUEL = stored_list[8]
            HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_CO = stored_list[9]
            HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_01 = stored_list[10]
            HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_02 = stored_list[11]
            HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_03 = stored_list[12]
            HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_AVG = stored_list[13]
            HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_A_PRI_AIR_FL = stored_list[14]
            HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_B_PRI_AIR_FL = stored_list[15]
            HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_C_PRI_AIR_FL = stored_list[16]
            HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_D_PRI_AIR_FL = stored_list[17]
            HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_TOT = stored_list[18]
            HIL_AL_UTKL_CPP_U1_BLR_01_ESP_A_OUTL_GAS_TEMP = stored_list[19]
            HIL_AL_UTKL_CPP_U1_BLR_01_ESP_B_OUTL_GAS_TEMP = stored_list[20]
            HIL_AL_UTKL_CPP_U1_BLR_01_ESP_OUTL_GAS_TEMP_AVG = stored_list[21]
            HIL_AL_UTKL_CPP_U1_BLR_01_SUPR_HTR_MN_STM_HDR_TEMP_02 = stored_list[22]
            HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_1_AIR_FL = stored_list[23]
            HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_2_AIR_FL = stored_list[24]
            HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_3_AIR_FL = stored_list[25]
            HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_4_AIR_FL = stored_list[26]
            HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_FAN_TOT = stored_list[27]
            CALC_LOSS_DUE_TO_H2_IN_FUEL = stored_list[28]
            PARTIAL_COMB_LOSS = stored_list[29]
            SteamGen = stored_list[30]
            TOTAL_UNBURNT_LOSS = stored_list[31]
            CALC_FINAL_EFF = stored_list[32]
            SENSIBLE_LOSS = stored_list[33]
            Tfg = stored_list[34]
            Excess_Air = stored_list[35]
            Theoritical_Air =stored_list[36]
            H2 =stored_list[37]
            GCV =stored_list[38]
            FDfanInlet =stored_list[39]
            ASH =stored_list[40]
            feedwaterfl =stored_list[41]
            SteamEnthalpy =stored_list[42]
            FeedwEnthalpy =stored_list[43]
            CO_APH_OUT =stored_list[44]
            TM =stored_list[45]
            C =stored_list[46]
            RejectQuality =stored_list[47]
            RejectGCV =stored_list[48]
            S =stored_list[49]
            N2 =stored_list[50]
            Radiation_Loss = stored_list[51]
            BottomAsh = stored_list[52]
            EspAsh = stored_list[53]
            Main_Steam_flow_trans_01 = stored_list[54]
            Blow_down_loss = stored_list[55]

            # Model input (need to maintaint the sequence)
            Model_inputs = [
                HIL_AL_UTKL_CPP_U1_BLR_01_CALC_TOT_LOSS,
                HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS,
                HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS,
                HIL_AL_UTKL_CPP_U1_BLR_01_HEAT_LOSS_IN_DRY_FLUE_GAS,
                HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS,
                HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_TOT,
                HIL_AL_UTKL_CPP_U1_BLR_01_CALC_SPECEFIC_COAL_CONS,
                HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_NOX,
                HIL_AL_UTKL_CPP_U1_BLR_01_ESP_OUTL_GAS_TEMP_AVG,
                HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_AIR,
                HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_AVG,
                HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_FUEL,
                HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_CO,
                HIL_AL_UTKL_CPP_U1_BLR_01_SUPR_HTR_MN_STM_HDR_TEMP_02,
                HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_FAN_TOT
            ]


            while True:
                steam_flow = float(stored_list[54])  # main steam flow
                print("steam:", steam_flow)
            
                if steam_flow < 120:
                    print("Skipping prediction: Steam flow is too low.")
                    time.sleep(60)  # Wait for 1 minute before retrying
                    continue  # Go to next iteration of while loop
            
                # If steam flow is sufficient, continue with prediction
                break
            
            Boiler_eff_prediction = self.model.predict([Model_inputs])[0]
            print("Before---->",Boiler_eff_prediction)
            print("original---->",CALC_FINAL_EFF)
            if Boiler_eff_prediction < CALC_FINAL_EFF:
                Boiler_eff_prediction = CALC_FINAL_EFF
                print("after---->",Boiler_eff_prediction)
        


            print("")
            print(f"Prediction: {Boiler_eff_prediction:.4f}")

            self.initialize_bounds() 
            
            res = minimize(self.objective, self.x0, method='SLSQP', bounds=self.bounds, constraints=self.cons)

            if not res.success:
                raise RuntimeError(res.message)

            x_opt = res.x
            efficiency = -res.fun
            print(efficiency,"=======Optimized efficiency======")
            _, loss_dict = self.calculate_losses(x_opt)
            
            print("\nOptimized Boiler Efficiency:", round(efficiency, 2), "%")


            physical_variables = {}
            losses = {}
            
            print("\n=== Physical Variables ===")
            for var, idx in self.var_index.items():
                val = x_opt[idx]
                physical_variables[var] = val
                # print(f"{var}: {val:.3f}")
            
            print("\n=== All Individual Losses ===")
            for name, val in loss_dict.items():
                losses[name] = val
            
            MILL_REJ = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS"]
            BLOW_DOWN = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS"]
            ESP_RAD = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS"]
            PA_FAN = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_PA_FAN_TOT"]
            COAL_CONS = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_CALC_SPECEFIC_COAL_CONS"]
            NOX = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_CHMNY_FLUE_GAS_NOX"]
            ESP_TEMP = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_ESP_OUTL_GAS_TEMP_AVG"]
            ECO_TEMP = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_ECO_FLUE_GAS_TEMP_AVG"]
            SUPR_HTR_HDR_1 = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_SUPR_HTR_MN_STM_HDR_TEMP_02"]
            WIND_BOX = physical_variables["HIL_AL_UTKL_CPP_U1_BLR_01_WIND_BOX_FAN_TOT"]
            
            Total_Loss = losses["Total Loss"]
            Dry_Flue_Gas_Loss = losses["L1: Dry Flue Gas Loss"]
            Moisture_in_Fuel = losses["L2: Moisture in Fuel"]
            H2_in_Fuel = losses["L3: H2 in Fuel"]
            H2O_in_Air = losses["L4: H2O in Air"]
            Partial_Combustion_Loss = losses["L5: Partial Combustion Loss"]
            Unburnt_ESP_Loss = losses["L6: Unburnt ESP Loss"]
            Unburnt_bottom_ash_loss = losses["L7: Unburnt bottom ash loss"]
            Total_Unburnt_Loss = losses["UL: Total Unburnt Loss"]
            Sensible_ESP_loss = losses["L8: Sensible ESP loss"]
            Sensible_Bottom_loss = losses["L9: Sensible Bottom loss"]
            Total_Sensible_Loss = losses["TL: Total Sensible Loss"]
            Radiation_Loss = losses["L10: Radiation Loss"]
            Blow_down_loss = losses['L11: BlowDown Loss']
            mill_reject_loss = losses['L12: Mill Rejection Loss']

            Act_list = [HIL_AL_UTKL_CPP_U1_BLR_01_CALC_TOT_LOSS,HIL_AL_UTKL_CPP_U1_BLR_01_CALC_MILL_REJCT_LOSS,
                        HIL_AL_UTKL_CPP_U1_BLR_01_CALC_BLW_DOWN_LOSS,HIL_AL_UTKL_CPP_U1_BLR_01_HEAT_LOSS_IN_DRY_FLUE_GAS,    HIL_AL_UTKL_CPP_U1_BLR_01_MNUL_ENTRY_ESP_RAD_LOSS,HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_AIR,
                        HIL_AL_UTKL_CPP_U1_BLR_01_CALC_LOSS_DUE_TO_H2O_IN_FUEL,CALC_LOSS_DUE_TO_H2_IN_FUEL,TOTAL_UNBURNT_LOSS,
                        PARTIAL_COMB_LOSS,SENSIBLE_LOSS]
        
            Pred_list = [Total_Loss,mill_reject_loss,Blow_down_loss,Dry_Flue_Gas_Loss,Radiation_Loss,H2O_in_Air,Moisture_in_Fuel,H2_in_Fuel,Total_Unburnt_Loss,
                         Partial_Combustion_Loss,Total_Sensible_Loss]  
            
            for i in range(len(Act_list)):
                print("before",Pred_list[i], Act_list[i])
                if Pred_list[i] > Act_list[i]:
                    Pred_list[i] = Act_list[i]
                    print("After",Pred_list[i] , Act_list[i])  
            Total_Loss = Pred_list[0]
            MILL_REJ = Pred_list[1]
            BLOW_DOWN = Pred_list[2]
            Dry_Flue_Gas_Loss = Pred_list[3]
            Radiation_Loss = Pred_list[4]
            H2O_in_Air = Pred_list[5]
            Moisture_in_Fuel = Pred_list[6]
            H2_in_Fuel = Pred_list[7]
            Total_Unburnt_Loss = Pred_list[8]
            Partial_Combustion_Loss = Pred_list[9]
            Total_Sensible_Loss = Pred_list[10]
            CALC_Total_Loss = (
                Dry_Flue_Gas_Loss + Radiation_Loss +
                H2O_in_Air + Moisture_in_Fuel + H2_in_Fuel + Total_Unburnt_Loss +
                Partial_Combustion_Loss + Total_Sensible_Loss
            )
            print("Total_Loss_after_optimization----->",CALC_Total_Loss)
            # Boiler_eff_prediction = Pred_list[11]

            predictions = {
            # "Total_Loss": Pred_list[0],
            "MILL_REJ": Pred_list[1],
            "BLOW_DOWN": Pred_list[2],
            "Dry_Flue_Gas_Loss": Pred_list[3],
            "Radiation_Loss": Pred_list[4],
            "H2O_in_Air": Pred_list[5],
            "Moisture_in_Fuel": Pred_list[6],
            "H2_in_Fuel": Pred_list[7],
            "Total_Unburnt_Loss": Pred_list[8],
            "Partial_Combustion_Loss": Pred_list[9],
            "Total_Sensible_Loss": Pred_list[10],
            "Boiler_eff_prediction":Boiler_eff_prediction
            }
            
            print(f"\n=== WRITING TO PI ===")
            self.write_predictions_to_pi(predictions)
            # time.sleep(30)


# if __name__ == "__main__":
#      utkal_B1 = SCCOptimizer()
#      utkal_B1.run_loop()
