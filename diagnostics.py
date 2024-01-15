
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


def missing_data_analysis():
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
def dataframe_summary():
    """ 
    Computes summary statistics
    
    Output: list containing all summary statistics
    """

    return 


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
    model_predictions()
    dataframe_summary()
    missing_data_analysis()
    execution_time()
    outdated_packages_list()
