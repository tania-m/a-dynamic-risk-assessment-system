
import pandas as pd
import numpy as np
import timeit
import os
import json

################## Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
print(f"Used dataset source folder: {dataset_csv_path}")

test_data_path = os.path.join(config['test_data_path'])
print(f"Test data folder: {dataset_csv_path}")

output_folder_path = config['output_folder_path']
print(f"Output folder: {output_folder_path}")


################## Function to get model predictions
def model_predictions():
    """ 
    Reads the deployed model and a test dataset, 
    calculate predictions
    
    Output: list containing all predictions
    """

    return 


def missing_data_analysis(df, columns_of_interest):
    """ 
    Calculates what percent of each column of the 
    ingested dataset consists of NA values
    
    Output:  list with the same number of elements as 
        the number of columns in the dataset 
        Each element of the list will be the 
        percent of NA values in a particular column
    """

    return


################## Function to get summary statistics
def dataframe_summary(df, columns_of_interest):
    """ 
    Computes summary statistics
    
    Output: table containing all summary statistics,
    with each row summarizing one predictor column
    """
    
    print("Dataframe overview statistics")
    print(df.describe())

    summary_statistics = []
    statistics_row = []
    summary_statistics.append(["column_name", "mean", "median", "std", "min", "max"])

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

    print(f"Running diagnostics using {ingested_dataset_name}")

    # model_predictions()
    dataframe_summary(ingested_df, predictors)
    # missing_data_analysis(ingested_df, predictors)
    # execution_time()
    # outdated_packages_list()
