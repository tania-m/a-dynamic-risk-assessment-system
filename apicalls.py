# Module name: app
"""
API calls to exposed endpoints
in app.py

Author: tania-m
Date: January 15th 2024
"""

import requests
import os
import json

################### Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
print(f"Test data folder: {test_data_path}")

model_path = os.path.join(config['output_model_path']) 
print(f"Model files source folder: {model_path}")

# specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8080"

# set headers: input as json, response as text for file serialization
headers = {"Content-type": "application/json", "Accept": "text/plain"}

# call each API endpoint and store the responses
print("## Calling /prediction endpoint")
predict_dataset_file = "testdata.csv"
predicat_dataset_path = os.path.join(test_data_path, predict_dataset_file)
predict_base_url = f"{URL}/prediction"
print(f"POST request to {predict_base_url} for dataset {predict_dataset_file}")
predict_response = requests.post(predict_base_url, json={"datafile": predicat_dataset_path}, headers=headers).text
print(f"POST request to {predict_base_url} for {predict_dataset_file}, response: {predict_response}")

scoring_url = f"{URL}/scoring"
print(f"## Calling /scoring endpoint at {scoring_url}")
scoring_response = requests.get(scoring_url, headers=headers).text
print(f"GET {scoring_url} response: {scoring_response}")

print("## Calling /summarystats endpoint")
predict_dataset_file = "testdata.csv"
predicat_dataset_path = os.path.join(test_data_path, predict_dataset_file)
summarystats_url = f"{URL}/summarystats"
summarystats_response = requests.get(summarystats_url, json={"datafile": predicat_dataset_path}, headers=headers).text
print(f"GET {summarystats_url} response for summary of {predicat_dataset_path}: {summarystats_response}")

print("## Calling /diagnostics endpoint")
diagnostics_url = f"{URL}/diagnostics"
diagnostics_response = requests.get(diagnostics_url, headers=headers).text
print(f"GET {diagnostics_url} response: {diagnostics_response}")

# combine all API responses into a (json) dictionnary
api_calls_responses = {
    "Predictions": predict_response,
    "Scoring": scoring_response,
    "Summary_Stats": summarystats_response,
    "Diagnostics": diagnostics_response,
}

# write the responses to your workspace
api_responses_txt_file = "apireturns.txt"
api_responses_file = "apireturns.json"

# Saving responses as text
api_response_txt_file_path = os.path.join(model_path, api_responses_txt_file)
with open(api_response_txt_file_path, "w") as written_file:
    written_file.write(json.dumps(api_calls_responses))
print(f"Saved API responses as text to {api_response_txt_file_path}")

# Saving results as JSON may make it easier to re-read the data later on
api_response_json_file_path = os.path.join(model_path, api_responses_file)
with open(api_response_json_file_path, "w") as written_file:
    written_file.write(json.dumps(api_calls_responses))
print(f"Saved API responses as JSON to {api_response_json_file_path}")
