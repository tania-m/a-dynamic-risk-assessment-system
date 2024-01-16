# Udacity project for MLOps: A dynamic risk assessment system

## Project overview

A company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue. They have a team of client managers who stay in contact with clients and try to convince them not to exit their contracts. However, the client management team is small, and they're not able to stay in close contact with all 10,000 clients.

The company needs us to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. If the model created and deploy is accurate, it will enable the client managers to contact the clients with the highest risk and avoid losing clients and revenue.

Because the industry is dynamic and constantly changing, and a model that was created a year or a month ago might not still be accurate today. Because of this, we need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. We'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that the company can get risk assessments that are as accurate as possible and minimize client attrition.

## Project steps

This project has 5 steps:

### Data ingestion

- Automatically check a database for new data that can be used for model training.
- Compile all training data to a training dataset and save it to persistent storage.
- Write metrics related to the completed data ingestion tasks to persistent storage.

### Training, scoring, and deploying

- Write scripts that train an ML model that predicts attrition risk, and score the model. 
- Write the model and the scoring metrics to persistent storage.

### Diagnostics

- Determine and save summary statistics related to a dataset.
- Time the performance of model training and scoring scripts.
- Check for dependency changes and package updates.

### Reporting

- Automatically generate plots and documents that report on model metrics.
- Provide an API endpoint that can return model predictions and metrics.

### Process Automation

- Write a script and cron job that automatically run all previous steps at regular intervals.
[Process Automation schema](https://github.com/tania-m/a-dynamic-risk-assessment-system/blob/main/images/pipeline-fullprocess.jpg)