
import pandas as pd
import numpy as np
import timeit
import pickle
import os
import json

from training import prepare_dataset
# We'll use the same function for dataset prep
# as we did in training
# Preparations done on the dataset:
# - The dataset's final column, "exited", is the target variable for predictions
# - The first column, "corporation", will not be used in modeling. 
# - The other three numeric columns will all be used as predictors in the model.

################## Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
print(f"Used dataset source folder: {dataset_csv_path}")

test_data_path = os.path.join(config['test_data_path'])
print(f"Test data folder: {dataset_csv_path}")

output_folder_path = config['output_folder_path']
print(f"Output folder: {output_folder_path}")

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
print(f"Prod deployment folder: {prod_deployment_path}")


################## Function to get model predictions
def model_predictions(df):
    """ 
    Reads the deployed model and a test dataset, 
    calculate predictions
    
    Output: list containing all predictions
    """
    
    print("Computing model predictions")
    
    print("Reading deployed model")
    deployed_model_name = "trainedmodel.pkl"
    deployed_model_path = os.path.join(prod_deployment_path, deployed_model_name)
    with open(deployed_model_path, "rb") as file:
        prod_model = pickle.load(file)
    print("Prod model loaded!")
    
    df_total_rows = df.shape[0]
    print(f"Dataset for prediction has {df_total_rows} rows")
    
    print("Preparing dataset for predictions")
    X, _ = prepare_dataset(df)
    print("Computing predictions")
    y_predicted = list(prod_model.predict(X))

    return y_predicted


def missing_data_analysis(df):
    """ 
    Calculates what percent of each column of the 
    ingested dataset consists of NA values
    
    Output:  table with total row count being 
        the same number of elements as 
        the number of columns in the dataset 
        Each row of the list will be the 
        column name and the percent of NA values 
        in that particular column (percent as float,
        between 0.0 and 1.0)
    """
    
    print("Computing dataframe missing data points")
    
    columns_of_interest = df.columns
    print(columns_of_interest)
    
    missing_data = []
    missing_data_row_header = [ "column_name", "missing_data_percentage"]
    missing_data.append(missing_data_row_header)
    
    total_rows = df.shape[0]
    
    if total_rows == 0:
        return []
    
    for column in columns_of_interest:
        missing_data_row = []
        missing_data_row.append(column)
        na_values = df[column].isna().sum()
        missing_data_row.append(na_values/total_rows)
        missing_data.append(missing_data_row)

    print(missing_data)
    return missing_data


################## Function to get summary statistics
def dataframe_summary(df, columns_of_interest):
    """ 
    Computes summary statistics
    
    Output: table containing all summary statistics,
    with each row summarizing one predictor column
    """
    
    print("Computing dataframe statistics")
    
    print("Dataframe overview statistics")
    print(df.describe())

    summary_statistics = []
    statistics_row_header = ["column_name", "mean", "median", "std", "min", "max"]
    summary_statistics.append(statistics_row_header)

    for column in columns_of_interest:
        statistics_row = []
        statistics_row.append(column)
        statistics_row.append(df[column].mean())
        statistics_row.append(df[column].median())
        statistics_row.append(df[column].std())
        statistics_row.append(df[column].min())
        statistics_row.append(df[column].max())
        
        summary_statistics.append(statistics_row)
    
    print("Extracted dataframe statistics")
    print(summary_statistics)
    return summary_statistics


################## Function to get timings
def execution_time():
    """ 
    Computes timing of training.py and ingestion.py
    
    Output: list of 2 timing values in seconds
    """

    return


################## Function to check dependencies
def outdated_packages_list():
    """ 
    Checks dependencies: current versions used and latest available
    
    Output: a table with three columns: the first column will show the name 
        of a Python module that you're using; the second column will show 
        the currently installed version of that Python module, and the third 
        column will show the most recent available version of that Python module
    """


if __name__ == '__main__':
    ingested_dataset_name =  "finaldata.csv"
    ingested_data_full_path = os.path.join(dataset_csv_path,ingested_dataset_name)
    print(f"Loading dataset {ingested_data_full_path}")
    ingested_df = pd.read_csv(ingested_data_full_path)

    predictors = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]

    test_data_name = "testdata.csv"
    test_data_full_path = os.path.join(test_data_path, test_data_name)
    test_dataframe = pd.read_csv(test_data_full_path)
    print(f"Getting prediction diagnostics using {test_data_full_path}")
    model_predictions(test_dataframe)
    
    print(f"Running diagnostics using {ingested_data_full_path}")
    dataframe_summary(ingested_df, predictors)
    missing_data_analysis(ingested_df)
    # execution_time()
    # outdated_packages_list()
