from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os

# model_predictions uses the dployed model
from diagnostics import model_predictions


############### Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
print(f"Used dataset source folder: {dataset_csv_path}")

test_data_path = os.path.join(config['test_data_path'])
print(f"Test data folder: {dataset_csv_path}")

model_path = os.path.join(config['output_model_path']) 
print(f"Model output folder: {model_path}")


############## Function for reporting
def score_model():
    """ 
    Scores the models:
    - calculates a confusion matrix using the test data and the deployed model
    - writes the confusion matrix to the workspace in output_model_path
    
    Output: list containing all predictions
    """
    
    confusion_matrix_filename = "confusionmatrix.png"
    confusion_matrix_full_path = os.path.join(model_path, confusion_matrix_filename)
    
    print("Preparing predictions to compute confusion matrix")
    test_data_name = "testdata.csv"
    test_data_full_path = os.path.join(test_data_path, test_data_name)
    print(f"Using test dataset from {test_data_full_path} for predictions")
    
    test_dataframe = pd.read_csv(test_data_full_path)
    # Get predictions
    y_predictions = model_predictions(test_dataframe)
    
    # Get "reality" from test dataset
    y_true = test_dataframe.pop("exited")
    
    # Get confusion matrix representation
    confusion_matrix = metrics.confusion_matrix(y_true, y_predictions)
    
    print("Raw confusion matrix: ")
    print(confusion_matrix)
    
    print("Plotting confusion matrix")
    # Using https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    visual_confusion_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    # Make sure the matrix has been plotted
    visual_confusion_matrix.plot()
    
    # Save the matrix to image file
    with open(confusion_matrix_full_path, "wb") as image_file:
        plt.savefig(image_file)
        print(f"Confusion matrix saved to {image_file}")


if __name__ == '__main__':
    score_model()
