# Module name: app
"""
Endpoints server

Author: tania-m
Date: January 15th 2024
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import json
import os

# From the data science pipeline
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
print(f"Used dataset source folder: {dataset_csv_path}")

prediction_model = None


####################### Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    """ 
    Prediction route
    """
    
    prediction_dataset_path = request.json.get("datafile")
    print(f"Prediction requested for dataset at path {prediction_dataset_path}")
    
    predict_df = pd.read_csv(prediction_dataset_path)
    print("Dataframe for predictions loaded")
    
    print("Getting predictions")
    y_pred = model_predictions(predict_df)
    
    return str(y_pred)


####################### Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():  
    """ 
    Scoring results route
    """       
    
    return str(score_model())


####################### Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
    """ 
    Summary stats route
    """ 
    
    summary_dataset_path = request.json.get("datafile")
    print(f"Summary requested for dataset at path {summary_dataset_path}")
    
    summary_df = pd.read_csv(summary_dataset_path)
    return str(dataframe_summary(summary_df))


####################### Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """ 
    Diagnostics route:
    - ingestion and training timing
    - percent NA values
    """ 
    
    timings = execution_time()
    outdated_packages = outdated_packages_list()
    
    results = [ ["timings", timings[0], timings[1]],
                ["outdated_packages", outdated_packages]]
    return str(results)

if __name__ == "__main__":
    host = "0.0.0.0"
    port=8080
    print("#############################################################") 
    print(f"Running app server on host {host} and port {port}")
    print("#############################################################")
    app.run(host=host, port=port, debug=True, threaded=True)
