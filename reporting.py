import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 


##############Function for reporting
def score_model():
    """ 
    Scores the models, creates plots:
    - calculates a confusion matrix using the test data and the deployed model
    - writes the confusion matrix to the workspace
    
    Output: list containing all predictions
    """


if __name__ == '__main__':
    score_model()
