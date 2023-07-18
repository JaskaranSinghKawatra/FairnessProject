from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
from app import celery
import tensorflow as tf
import numpy as np
import time
from app.main.training_models import TrainingModels

@celery.task(bind=True)
def run_training(self, file_path, target_variable, sensitive_attribute, model_type, fairness_definition):
    trainer =  TrainingModels(file_path, target_variable, sensitive_attribute)
    return trainer.train_model(model_type, fairness_definition)



# def train_model(self, file_path, target_variable, sensitive_attribute, model_type, fairness_definition):
#     print("Train model task is being executed.")
#     start_time = time.time()
#     # Load the dataset
#     df = pd.read_csv(file_path)

#     # if model_type == 'logistic_regression':
#     #     if fairness_definition == 'demographic_parity':
#     #         # Demographic Parity Logistic Regression Model
#     #         return train_logistic_regression_demographic_parity(df, target_variable, sensitive_attribute)



# @celery.task(bind=True)
# def train_model(self, file_path, target_variable, sensitive_attribute):
#     print("Train model task is being executed.")
#     # Load the dataset
#     df = pd.read_csv(file_path)

#     # Separate the features and target
#     X = df.drop(columns=[target_variable, sensitive_attribute])
#     y = df[target_variable]

#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize and train the model
#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     # Make predictions
#     predictions = model.predict(X_test)

#     # Calculate the accuracy of the model
#     model_accuracy = model.score(X_test, y_test)

#     # Calculate the AUC score
#     auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

#     return predictions.tolist(), model_accuracy, auc_score










