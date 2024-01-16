# Module name: fullprocess
"""
Automate data science pipeline process

Author: tania-m
Date: January 15th 2024
"""

import json
import os
import sys
import pandas as pd

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
print(f"Input folder: {input_folder_path}")

dataset_csv_path = os.path.join(config['output_folder_path']) 
print(f"Dataset used for training source folder: {dataset_csv_path}")

model_path = os.path.join(config['output_model_path']) 
print(f"Model output folder: {model_path}")

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
print(f"Prod deployment folder: {prod_deployment_path}")

print('################################################')
################## Check and read new data
# first, read ingestedfiles.txt
ingested_files_name = "ingestedfiles.txt"
ingested_files_full_path = os.path.join(dataset_csv_path, ingested_files_name)
print(f"Load ingested files details from {ingested_files_full_path}")
with open(ingested_files_full_path, "r") as inggested_files_list:
    ingestedfiles = inggested_files_list.read()

# split file on lines
all_ingested_files = ingestedfiles.split("\n")

# build list of all ingested files
ingestion_list = []
for ingestion_item in all_ingested_files:
    split_line = ingestion_item.split(" - ")
    
    # if we have a non empty line from the ingestion logs
    if len(split_line) == 2:
        # append filename to list 
        ingestion_list.append(split_line[1])

# list all files in source folder
source_files_list = os.listdir(input_folder_path)

# we only look at files that may be in source_files_list but not in ingestion_list
# we ignore the other way around if ingestion_list has now more files than source_files_list
list_difference = list(set(source_files_list) - set(ingestion_list))

################## Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if len(list_difference) > 0:
    print(f"Found new data files. Running new ingestion for {str(list_difference)}")
    ingestion.merge_multiple_dataframe()
else:
    print("No new data files. Pipeline execution ending")
    sys.exit(0)

################## Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
print("Training model based on new data")
training.train_model()
new_model_f1_score = scoring.score_model()
print(f"New model has f1_score of {new_model_f1_score}")

deployed_model_score = "latestscore.txt"
deployed_model_score_full_path = os.path.join(prod_deployment_path, deployed_model_score)
print(f"Read current prod model score from {deployed_model_score_full_path}")
with open(deployed_model_score_full_path, "r") as f1_score_file:
    current_f1_score = str(f1_score_file.read())
print(f"Current prod model has f1_score {current_f1_score}")

################## Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if True or new_model_f1_score > current_f1_score:
    # Higher f1 score is better
    print("Model drift detected")
else:
    print("No model drift. Pipeline execution ending")
    sys.exit(0)

##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
print("Deploying new model to prod")
deployment.move_deployment_files()

################## Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
print("Running reporting for the newly deployed model")
reporting.score_model()

print("Running diagnostics for the newly deployed model, using newly ingested data")
ingested_dataset_name =  "finaldata.csv"
ingested_data_full_path = os.path.join(dataset_csv_path,ingested_dataset_name)
print(f"Loading dataset {ingested_data_full_path}")
ingested_df = pd.read_csv(ingested_data_full_path)

print(f"Running diagnostics using {ingested_data_full_path}")
diagnostics.dataframe_summary(ingested_df)
diagnostics.missing_data_analysis(ingested_df)
diagnostics.execution_time()
diagnostics.outdated_packages_list()
print("Finished diagnostics for new model!")
