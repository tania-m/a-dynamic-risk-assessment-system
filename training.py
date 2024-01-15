# Module name: training
"""
Model training for risk assessment data science pipeline

Uses config from config.json
Ingestion record will be in output_folder_path/ingestedfiles.txt

Author: tania-m
Date: January 15th 2024
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

################### Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


def prepare_for_training(df):
    """ 
    Prepare the dataset for training:
    - The dataset's final column, "exited", is the target variable for predictions
    - The first column, "corporation", will not be used in modeling. 
    - The other three numeric columns will all be used as predictors in the model.
    """
    
    # Remove first column and keep only predictors
    predictors = df.loc[:, ["lastmonth_activity", 
                            "lastyear_activity",
                            "number_of_employees"]]
    
    # Separate "exited", as the target variable for predictions
    # No encoding done here, as existed is already numerical 0/1
    predicted = df["exited"]
    
    # Use more common var names
    X = predictors
    y = predicted
    
    return X, y


################# Function for training the model
def train_model():
    
    # Using this logistic regression for training
    print("Configuring logistic regression used for training")
    
    # In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme 
    # if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss 
    # if the ‘multi_class’ option is set to ‘multinomial’. 
    # (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, 
    #  ‘sag’, ‘saga’ and ‘newton-cg’ solvers.)
    logistic_regression_model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='ovr', # multiclass changed to ovr because of used liblinear solver
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear', 
        tol=0.0001, 
        verbose=0,
        warm_start=False)
    
    # Loading data and fitting the logistic regression to the data
    dataset_name = "finaldata.csv"
    input_data_path = os.path.join(dataset_csv_path, dataset_name)
    print(f"Reading ingested data {dataset_name} from {input_data_path}")
    dataframe = pd.read_csv(input_data_path)
    
    print("Preparing dataset for training")
    X, y = prepare_for_training(dataframe)
    
    print("train_test_split")
    test_size_proportion = 0.20 # proportion of the dataset to include in the test split
    random_state_seed = 42 # to make runs reproduceable
    x_train, x_test, y_train, y_test = train_test_split(X, y,  
                                                        test_size=test_size_proportion,
                                                        random_state=random_state_seed)
    
    print("Fitting logistic regression model")
    logistic_regression_model.fit(x_train, y_train)
    print("Fitting logistic regression model DONE!")
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    model_name = "trainedmodel.pkl"
    model_full_path = os.path.join(model_path, model_name)
    print(f"Saving trained model to {model_path} in pickle format to {model_full_path}")
    try:
        pickle.dump(logistic_regression_model, open(model_full_path, "wb"))
    except:
        print("Target folder doesn't seem to existing. Creating it...")
        os.mkdir(model_path)
        pickle.dump(logistic_regression_model, open(model_full_path, "wb"))
    print(f"Model saved to {model_full_path}!")


if __name__ == "__main__":
    train_model()