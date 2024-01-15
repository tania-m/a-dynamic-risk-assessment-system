# Module name: scoring
"""
Model scoring for risk assessment data science pipeline

Uses config from config.json
Scores model trained by training.py

Author: tania-m
Date: January 15th 2024
"""

import pandas as pd
import pickle
import os
from sklearn import metrics
import json

from training import prepare_dataset
# We'll use the same function for dataset prep
# as we did in training
# Preparations done on the dataset:
# - The dataset's final column, "exited", is the target variable for predictions
# - The first column, "corporation", will not be used in modeling. 
# - The other three numeric columns will all be used as predictors in the model.


################# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
print(f"Dataset used source folder: {dataset_csv_path}")

model_path = os.path.join(config['output_model_path'])
print(f"Trained model source folder: {model_path}")

test_data_path = os.path.join(config['test_data_path']) 
print(f"Test data folder: {dataset_csv_path}")


################# Function for model scoring
def score_model():
    """ 
    Score model:
    take a trained model, 
    load test data, 
    and calculate an F1 score for the model relative to the test data
    
    Output: writes the resulting f1_score to `latestscore.txt`
    """
    
    model_name = "trainedmodel.pkl"
    trained_model_full_path = os.path.join(model_path, model_name)
    print(f"Loading trained model from: {trained_model_full_path}")
    with open(trained_model_full_path, "rb") as model_to_load:
        model = pickle.load(model_to_load)
    print("Loaded trained model from!")
    
    test_data_name = "testdata.csv"
    test_data_full_path = os.path.join(test_data_path, test_data_name)
    print(f"Loading test data from {test_data_full_path} into dataframe")
    test_dataframe = pd.read_csv(test_data_full_path)
    
    print("Preparing dataset")
    X, y = prepare_dataset(test_dataframe)
    
    print("Scoring trained model")
    y_predictions = model.predict(X)
    f1_score = metrics.f1_score(y, y_predictions)
    print(f"f1_score is {f1_score}")
    
    scoring_result_filename = "latestscore.txt"
    print(f"Saving scoring result to local file {scoring_result_filename}")
    scoring_results_path = os.path.join(model_path, scoring_result_filename)
    with open(scoring_results_path, "w") as result_file:
        try:
            result_file.write(str(f1_score))
        except FileNotFoundError:
            # Should not happen currently since we loaded the model from there,
            # but in case we ever change the scoring file's location...
            print("Target folder doesn't seem to existing. Creating it...")
            os.mkdir(model_path)
            result_file.write(str(f1_score))
    print(f"Scoring result saved to {scoring_result_filename}")


if __name__ == "__main__":
    score_model()