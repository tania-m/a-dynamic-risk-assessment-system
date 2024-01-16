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

# specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8080"

# set headers: input as json, response as text for file serialization
headers = {"Content-type": "application/json", "Accept": "text/plain"}

# call each API endpoint and store the responses
# print("Calling /predict endpoint")
# predict_dataset_file = "testdata.csv"
# predicat_dataset_path = os.path.join(test_data_path, predict_dataset_file)
# predict_base_url = f"{URL}/predict"

# print(f"POST request to {predict_base_url}")
# predict_response = requests.post(predict_base_url, json={"datafile": predict_dataset_file}, headers=headers)
# print(predict_response)

scoring_url = f"{URL}/scoring"
print(f"Calling /scoring endpoint at {scoring_url}")
scoring_response = requests.get(scoring_url, headers=headers).text
print(f"{scoring_url} response: {scoring_response}")

# print("Calling /summarystats endpoint")
# summarystats_url = f"{URL}/summarystats"
# summarystats_response = requests.get(summarystats_url, headers=headers)

# print("Calling /diagnostics endpoint")
# diagnostics_url = f"{URL}/diagnostics"
# diagnostics_response = requests.get(diagnostics_url, headers=headers)

# # combine all API responses into a (json) dictionnary
# responses = {
#     "Predictions": predict_response,
#     "Scoring": scoring_response,
#     "Summary_Stats": summarystats_response,
#     "Diagnostics": diagnostics_response,
# }

# write the responses to your workspace
# print(responses)


