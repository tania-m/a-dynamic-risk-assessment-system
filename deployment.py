# Module name: deployment
"""
Model deployment for risk assessment data science pipeline

Uses model trained by model.py, 
F1 score generated by scoring.py,
ingestion record generated by ingestion.py

Uses config from config.json

Author: tania-m
Date: January 15th 2024
"""

import os
import shutil
import json


################## Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'])
print(f"Dataset used source folder: {dataset_csv_path}")

model_path = os.path.join(config['output_model_path']) 
print(f"Model files source folder: {model_path}")

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
print(f"Prod deployment folder: {prod_deployment_path}")


#################### Function for deployment
def move_deployment_files():
    """ 
    copy the latest model as pickle file, 
    the latestscore.txt value, 
    and the ingestfiles.txt file 
    into the deployment directory
    
    Output: writes the resulting f1_score to `latestscore.txt`
    """
    
    ingestion_report_name = "ingestedfiles.txt"
    ingestion_report_source_path = os.path.join(dataset_csv_path, ingestion_report_name)
    ingestion_report_target_path = os.path.join(prod_deployment_path, ingestion_report_name)
    print(f"Copying {ingestion_report_name} from {ingestion_report_source_path} to the deployment directory {prod_deployment_path}")
    try:
        shutil.copy(ingestion_report_source_path, ingestion_report_target_path)
    except FileNotFoundError:
        # prod_deployment_path folder may not be here at first run
        print("Target folder doesn't seem to existing. Creating it...")
        os.mkdir(prod_deployment_path)
        shutil.copy(ingestion_report_source_path, ingestion_report_target_path)
        
    model_files = os.listdir(model_path)
    for data_file in model_files:
        source_path = os.path.join(model_path, data_file)
        target_path = os.path.join(prod_deployment_path, data_file)
        # prod_deployment_path folder should exist now
        # if this is the first run, the folder got created above
        # if the folder is not here anymore, something is really off
        # and we'll let it crash
        print(f"Copying {source_path} to {target_path}")
        shutil.copy(source_path, target_path)


if __name__ == "__main__":
    move_deployment_files()
