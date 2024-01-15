# Module name: ingestion
"""
Data ingestion for risk assessment data science pipeline

Uses config from config.json
Ingestion record will be in output_folder_path/ingestedfiles.txt

Author: tania-m
Date: January 15th 2024
"""

import pandas as pd
import datetime
import os
import json
from datetime import datetime


############# Load config.json and get input and output paths
with open('config.json','r') as f:
    print("Loading configuration from config.json")
    config = json.load(f) 

input_folder_path = config['input_folder_path']
print(f"Input folder: {input_folder_path}")

output_folder_path = config['output_folder_path']
print(f"Output folder: {output_folder_path}")


############# Function for data ingestion
def merge_multiple_dataframe():
    """ 
    merge multiple datasets to one,
    write output to file `finaldata.csv`
    input: None
    output: file written to disk with dataset content, finaldata.csv
    """
    output_file_name = "finaldata.csv"
    ingestion_record_name = "ingestedfiles.txt"
    
    print(f"Getting list of all files to ingest at root of {input_folder_path}")
    files_to_ingest = os.listdir(input_folder_path)
    
    ingestion_record = []
    loaded_data = pd.DataFrame()
    
    for data_file in files_to_ingest:
        data_file_path = os.path.join(input_folder_path, data_file)
        print(f"Ingesting content of file {data_file} at {data_file_path}")
        
        temp_df = pd.read_csv(data_file_path)
        print("Data loaded into temporary dataframe")
        
        loaded_data = pd.concat([loaded_data, temp_df], axis=0)
        print("loaded data merged into loaded_data dataframe")
        
        ingestion_record.append([datetime.now(), data_file])
        
    print("Removing duplicates")
    loaded_data = loaded_data.drop_duplicates(ignore_index=True)
    
    output_file_path = os.path.join(output_folder_path, output_file_name)
    print(f"Writing result dataset to local filesystem at {output_file_path}")
    try:
        loaded_data.to_csv(output_file_path, index=False)
    except FileNotFoundError:
        print("Target folder doesn't seem to existing. Creating it...")
        os.mkdir(output_folder_path)
        loaded_data.to_csv(output_file_path, index=False)
    print(f"Dataset written to {output_file_path}")
    
    # Creating ingestion record
    ingestion_record_path = os.path.join(output_folder_path, ingestion_record_name)
    with open(ingestion_record_path, 'w') as f:
        for record in ingestion_record:
            f.write(f"{str(record[0])} - File {record[1]} ingested\n")
    print(f"Ingestion record written to {ingestion_record_path}")


#############
# About the data in finaldata.csv:
    
# The data in finaldata.csv represents records of corporations, 
# their characteristics, and their historical attrition records. 
# One row represents a hypothetical corporation. 

# There are five columns in the dataset:
# "corporation", which contains four-character 
#     abbreviations for names of corporations
# "lastmonth_activity", which contains the level of activity 
#     associated with each corporation over the previous month
# "lastyear_activity", which contains the level of activity 
#     associated with each corporation over the previous year
# "number_of_employees", which contains the number of employees 
#     who work for the corporation
# "exited", which contains a record of whether the corporation 
#     exited their contract (1 indicates that the corporation exited, 
#                            and 0 indicates that the corporation did not exit)
#
# The dataset's final column, "exited", is the target variable for predictions
# from this data science pipeline
#############

if __name__ == '__main__':
    merge_multiple_dataframe()
