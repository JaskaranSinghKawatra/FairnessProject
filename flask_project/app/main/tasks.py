from app import celery
from app.FairModels.training_models import TrainingModels


# @celery.task(bind=True)
# def run_training(self, file_path, target_variable, sensitive_attribute, model_type, fairness_definition):
#     trainer =  TrainingModels(file_path, target_variable, sensitive_attribute)
#     return trainer.train_model(model_type, fairness_definition)

@celery.task(bind=True)
def run_training(self, file_path, 
                 target_variable, 
                 sensitive_attribute, 
                 model_type, 
                 fairness_definition, 
                 learning_rate, 
                 lambda_fairness, 
                 num_epochs, 
                 batch_size):
    trainer =  TrainingModels(file_path, target_variable, sensitive_attribute)
    trainer.set_hyperparameters(learning_rate, lambda_fairness, num_epochs, batch_size)
    return trainer.train_model(model_type, fairness_definition)


# def multiple_models(self, file_path, target_variable, sensitive_attribute, model_type, fairness_definition):
#     learning_rate_grid = [0.01, 0.1, 1]
#     lambda_fairness_grid = [0.1, 1, 10]
#     hyperparameters = list(product(learning_rate_grid, lambda_fairness_grid))

#     args_list = [(file_path, target_variable, sensitive_attribute, model_type, fairness_definition, lr, lf) for lr, lf in hyperparameters]
#     with Pool() as pool:
#         results = pool.map(run_training, args_list)
#     results_df = pd.DataFrame(results)
#     print(results_df)
#     return results_df




#     # Create a multiprocessing pool and map the function to the hyperparameter
#     with mp.Pool(mp.cpu_count()) as pool:
#         results = pool.starmap(train_model, args_list)
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










