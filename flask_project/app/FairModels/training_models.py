
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from .logistic_regression_demographic_parity import logistic_regression_demographic_parity
# from FairModels.logistic_regression_disparate_impact import logistic_regression_disparate_impact
# from FairModels.logistic_regression_equal_opportunity import logistic_regression_equal_opportunity
# from FairModels.logistic_regression_equal_odds import logistic_regression_equal_odds
# from FairModels.logistic_regression_group_unawareness import logistic_regression_group_unawareness
# from FairModels.logistic_regression_negative_predictive_value_parity import logistic_regression_negative_predictive_value_parity
# from FairModels.logistic_regression_positive_predictive_value_parity import logistic_regression_positive_predictive_value_parity
# from FairModels.logistic_regression_false_positive_ratio_parity import logistic_regression_false_positive_ratio_parity
# from FairModels.perceptron_demographic_parity import perceptron_demographic_parity
# from FairModels.perceptron_disparate_impact import perceptron_disparate_impact
# from FairModels.perceptron_equal_opportunity import perceptron_equal_opportunity
# from FairModels.perceptron_equal_odds import perceptron_equal_odds
# from FairModels.perceptron_group_unawareness import perceptron_group_unawareness
# from FairModels.perceptron_negative_predictive_value_parity import perceptron_negative_predictive_value_parity
# from FairModels.perceptron_positive_predictive_value_parity import perceptron_positive_predictive_value_parity
# from FairModels.perceptron_false_positive_ratio_parity import perceptron_false_positive_ratio_parity
# from FairModels.neural_network_demographic_parity import neural_network_demographic_parity
# from FairModels.neural_network_disparate_impact import neural_network_disparate_impact
# from FairModels.neural_network_equal_opportunity import neural_network_equal_opportunity
# from FairModels.neural_network_equal_odds import neural_network_equal_odds
# from FairModels.neural_network_group_unawareness import neural_network_group_unawareness
# from FairModels.neural_network_negative_predictive_value_parity import neural_network_negative_predictive_value_parity
# from FairModels.neural_network_positive_predictive_value_parity import neural_network_positive_predictive_value_parity
# from FairModels.neural_network_false_positive_ratio_parity import neural_network_false_positive_ratio_parity


class TrainingModels:
    def __init__(self, df_path, target_variable, sensitive_attribute):
        self.df_path = df_path
        self.target_variable = target_variable
        self.sensitive_attribute = sensitive_attribute
        
    def set_hyperparameters(self, learning_rate, lambda_fairness):
        self.learning_rate = learning_rate
        self.lambda_fairness = lambda_fairness

    def get_hyperparameters(self):
        return self.learning_rate, self.lambda_fairness

    def train_model(self, model_type, fairness_definition):

        if model_type == 'logistic_regression':
            if fairness_definition == 'demographic_parity':
                return logistic_regression_demographic_parity(self.df_path, self.target_variable, self.sensitive_attribute, self.learning_rate, self.lambda_fairness)
            # elif fairness_definition == 'group_unawareness':
            #     return self.logistic_regression_group_unawareness()
            # elif fairness_definition == 'disparate_impact':
            #     return self.logistic_regression_disparate_impact()
            # elif fairness_definition == 'equal_opportunity':
            #     return self.logistic_regression_equal_opportunity()
            # elif fairness_definition == 'equal_odds':
            #     return self.logistic_regression_equal_odds()
            # elif fairness_definition == 'positive_predictive_value_parity':
            #     return self.logistic_regression_positive_predictive_value_parity()
            # elif fairness_definition == 'false_positive_ratio_parity':
            #     return self.logistic_regression_false_positive_ratio_parity()
            # elif fairness_definition == 'negative_predictive_value_parity':
            #     return self.logistic_regression_negative_predictive_value_parity()
            else:
                return "Error: Invalid fairness definition."
        # elif model_type == 'perceptron':
        #     if fairness_definition == 'group_unawareness':
        #         return self.perceptron_group_unawareness()
        #     elif fairness_definition == 'demographic_parity':
        #         return self.perceptron_demographic_parity()
        #     elif fairness_definition == 'disparate_impact':
        #         return self.perceptron_disparate_impact()
        #     elif fairness_definition == 'equal_opportunity':
        #         return self.perceptron_equal_opportunity()
        #     elif fairness_definition == 'equal_odds':
        #         return self.perceptron_equal_odds()
        #     elif fairness_definition == 'positive_predictive_value_parity':
        #         return self.perceptron_positive_predictive_value_parity()
        #     elif fairness_definition == 'false_positive_ratio_parity':
        #         return self.perceptron_false_positive_ratio_parity()
        #     elif fairness_definition == 'negative_predictive_value_parity':
        #         return self.perceptron_negative_predictive_value_parity()
        #     else:
        #         return "Error: Invalid fairness definition."
        # elif model_type == 'neural_network':
        #     if fairness_definition == 'group_unawareness':
        #         return self.neural_network_group_unawareness()
        #     elif fairness_definition == 'demographic_parity':
        #         return self.neural_network_demographic_parity()
        #     elif fairness_definition == 'disparate_impact':
        #         return self.neural_network_disparate_impact()
        #     elif fairness_definition == 'equal_opportunity':
        #         return self.neural_network_equal_opportunity()
        #     elif fairness_definition == 'equal_odds':
        #         return self.neural_network_equal_odds()
        #     elif fairness_definition == 'positive_predictive_value_parity':
        #         return self.neural_network_positive_predictive_value_parity()
        #     elif fairness_definition == 'false_positive_ratio_parity':
        #         return self.neural_network_false_positive_ratio_parity()
        #     elif fairness_definition == 'negative_predictive_value_parity':
        #         return self.neural_network_negative_predictive_value_parity()
        #     else:
        #         return "Error: Invalid fairness definition."
        else:
            return "Error: Invalid model type."
        



    # def logistic_regression_demographic_parity(self):
    #     start_time = time.time()

    #     # for column in self.df.columns:
    #     #     if self.df[column].dtype == 'object':
    #     #         self.df[column] = self.df[column].astype(str)

    #     print("Training logistic regression model with demographic parity fairness definition")

    #     # Separate numerical and categorical features
    #     numerical_cols = self.df.select_dtypes(include=[np.number]).columns
    #     categorical_cols = self.df.select_dtypes(include=['object','string']).columns

    #     # Get sets of unique counts for each type of column
    #     num_unique_counts = set(self.df[numerical_cols].nunique())
    #     cat_unique_counts = set(self.df[categorical_cols].nunique())

    #     # Ensure target variable and sensitive attribute are not included in the numerical and categorical columns
    #     numerical_cols = numerical_cols.drop(self.target_variable) if self.target_variable in numerical_cols else numerical_cols
    #     numerical_cols = numerical_cols.drop(self.sensitive_attribute) if self.sensitive_attribute in numerical_cols else numerical_cols
    #     categorical_cols_preprocessing = categorical_cols.drop(self.target_variable, self.sensitive_attribute)
        
    #     # Identify numerical columns to drop based on matching unique counts
    #     cols_to_drop = [col for col in numerical_cols if self.df[col].nunique() in cat_unique_counts]

    #     # Update numerical columns by dropping identified columns
    #     numerical_cols = numerical_cols.drop(cols_to_drop)

    #     preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('num', StandardScaler(), numerical_cols),
    #             ('cat', OneHotEncoder(), categorical_cols_preprocessing)])
        
    #     le = LabelEncoder()
        
    #     for column in self.df.columns:
    #         if self.df[column].dtype == 'object':
    #             self.df[column] = self.df[column].astype(str)



    #     # Separate the features and target

    #     X = preprocessor.fit_transform(self.df.drop([self.target_variable, self.sensitive_attribute], axis=1))
    #     y = le.fit_transform(self.df[self.target_variable]).astype(np.float32)    

    #     # X = self.df.drop(columns=[self.target_variable, self.sensitive_attribute]).values.astype(np.float32)
    #     # y = self.df[self.target_variable].values.astype(np.float32)
        
    #     # Get unique values for the sensitive attribute and create a mapping
    #     unique_values = self.df[self.sensitive_attribute].unique()
    #     mapping_dict = {value: i for i, value in enumerate(unique_values)}
        
    #     # Apply the mapping to the column and convert to integers
    #     A = self.df[self.sensitive_attribute].map(mapping_dict).values.astype(np.int32)

    #     # One-hot encode the sensitive attribute
    #     one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #     A_one_hot = one_hot_encoder.fit_transform(A.reshape(-1, 1))
    #     # Split the data into training and test sets
    #     X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A_one_hot, test_size=0.2, random_state=42)
        
    #     # Convert sparse matrices to dense matrices
    #     X_train = X_train.toarray() if scipy.sparse.issparse(X_train) else X_train
    #     X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test

    #     X_train = tf.cast(X_train, tf.float32)
    #     X_test = tf.cast(X_test, tf.float32)
    #     y_test = tf.cast(y_test, tf.float32)
    #     A_test = tf.cast(A_test, tf.float32)

    #     # Initialize the weights and bias
    #     w = tf.Variable(tf.random.normal([X.shape[1], 1], dtype = tf.float32))
    #     b = tf.Variable(tf.zeros([1], dtype = tf.float32))

    #     # Define the logistic regression model
    #     def model(X):
    #         X = tf.cast(X, tf.float32)
    #         return tf.keras.activations.sigmoid(tf.matmul(X, w) + b)

    #     # Define the fairness penalty
    #     def fairness_penalty(A_one_hot, predictions):
    #         A_one_hot = tf.cast(A_one_hot, tf.float32)
    #         predictions = tf.cast(predictions, tf.float32)
    #         group_predictions = tf.matmul(tf.transpose(A_one_hot), predictions)
    #         group_counts = tf.reduce_sum(A_one_hot, axis=0)
    #         group_averages = group_predictions / group_counts[:, tf.newaxis]
    #         # Check for NaNs
    #         # tf.debugging.check_numerics(group_predictions, message='Debugging: NaN found in group_predictions')
    #         # tf.debugging.check_numerics(group_counts, message='Debugging: NaN found in group_counts')
    #         # tf.debugging.check_numerics(group_averages, message='Debugging: NaN found in group_averages')
    #         max_diff = tf.reduce_max(group_averages) - tf.reduce_min(group_averages)
    #         return max_diff

    #         # group_0_mask = tf.equal(A, 0)
    #         # group_1_mask = tf.equal(A, 1)
    #         # predictions_0 = tf.boolean_mask(predictions, group_0_mask)
    #         # predictions_1 = tf.boolean_mask(predictions, group_1_mask)
    #         # return tf.abs(tf.reduce_mean(predictions_0) - tf.reduce_mean(predictions_1))

    #     # Define the custom loss function
    #     def loss_fn(y_true, y_pred, A_one_hot):
    #         y_true = tf.cast(y_true, tf.float32)
    #         y_pred = tf.cast(y_pred, tf.float32)
    #         A_one_hot = tf.cast(A_one_hot, tf.float32)
    #         # lambda_fairness = tf.cast(lambda_fairness, tf.float32)
    #         epsilon = 1e-7
    #         log_loss = -tf.reduce_mean(y_true * tf.math.log(y_pred + epsilon) + (1 - y_true) * tf.math.log(1 - y_pred + epsilon))

    #         # Check for NaNs
    #         #tf.debugging.check_numerics(log_loss, message='Debugging: NaN found in log_loss')
    #         fairness_loss = fairness_penalty(A_one_hot, y_pred)
    #         # Check for NaNs
    #         #tf.debugging.check_numerics(fairness_loss, message='Debugging: NaN found in fairness_loss')
    #         return log_loss + self.lambda_fairness * fairness_loss

    #     # Define the optimizer
    #     optimizer = tf.optimizers.Adam(self.learning_rate)

    #     # Create a tf.data.Dataset object
    #     train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train, A_train))

    #     # Shuffle and batch the data
    #     batch_size = 32
    #     train_data = train_data.shuffle(buffer_size=1024).batch(batch_size)

    #     # Initialize lists to store the values for each iteration
    #     loss_values = []
    #     fairness_values = []
    #     accuracy_values = []

    #     with tf.device('/GPU:0'):

    #     # Train the model
    #         for _ in range(100):  # number of training iterations
    #             predictions_list = []
    #             for batch_x, batch_y, batch_a in train_data:
    #                 with tf.GradientTape() as tape:
    #                     # X_train_tf = tf.cast(X_train, tf.float64)
    #                     predictions = model(batch_x)
    #                     predictions_list.append(predictions)

    #                     # print(f"Min prediction: {tf.reduce_min(predictions).numpy()}, Max prediction: {tf.reduce_max(predictions).numpy()}")


    #                     # predictions = model(X_train)
    #                     loss = loss_fn(batch_y, predictions, batch_a)  # adjust lambda_fairness as needed

                        

    #                     # Check for NaNs in predictions and loss
    #                     # tf.debugging.check_numerics(predictions, message='Debugging: NaN found in predictions')
    #                     # tf.debugging.check_numerics(loss, message='Debugging: NaN found in loss')

    #             grads = tape.gradient(loss, [w, b])
    #             optimizer.apply_gradients(zip(grads, [w, b]))

    #         # Calculate the fairness and accuracy for this iteration
    #         current_accuracy = np.mean((predictions.numpy() > 0.5) == batch_y)
    #         current_fairness = fairness_penalty(batch_a, predictions).numpy()

    #         # After the end of each epoch, concatenate the predictions for all batches
    #         epoch_predictions = tf.concat(predictions_list, axis=0)

    #         # At the end of each epoch, calculate the predictive rates and fairness metrics
    #         group_predictive_rates = calculate_predictive_rates(y_true, y_pred, A_one_hot)
    #         demographic_parity_difference, demographic_parity_ratio = calculate_fairness_metrics(y_train, predictions, A_train)
    #         predictions_list.clear()
            
    #         # Append the loss, accuracy, and fairness values for this epoch
    #         loss_values.append(loss.numpy())
    #         accuracy_values.append(current_accuracy)
    #         fairness_values.append(current_fairness)

    #     # Make predictions
    #     predictions_test = model(X_test)

    #     # Calculate the accuracy of the model
    #     model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)

    #     # Calculate the AUC score
    #     auc_score = roc_auc_score(y_test, predictions_test.numpy())

    #     # Calculate the fairness metric (demographic parity)
    #     fairness_metric = fairness_penalty(A_test, predictions_test).numpy()

    #     end_time = time.time()  # End measuring execution time
    #     execution_time = end_time - start_time
    #     print("Execution time: {:.2f} seconds".format(execution_time))

    #     return predictions_test.numpy().tolist(), model_accuracy, auc_score

    # def logistic_regression_disparate_impact(self):
    #     start_time = time.time()
    #     X = self.df.drop(columns=[self.target_variable, self.sensitive_attribute]).values
    #     y = self.df[self.target_variable].values

    #     # Get unique values for the sensitive attribute and create a mapping
    #     unique_values = self.df[self.sensitive_attribute].unique()
    #     mapping_dict = {value: i for i, value in enumerate(unique_values)}
    #     num_groups = len(unique_values)
    #     print("Number of Groups: ", num_groups)
        
    #     # Apply the mapping to the column and convert to integers
    #     A = self.df[self.sensitive_attribute].map(mapping_dict).values.astype(np.int32)

    #     # One-hot encode the sensitive attribute
    #     one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #     A_one_hot = one_hot_encoder.fit_transform(A.reshape(-1, 1))

    #     # Split the data into training and test sets
    #     X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A_one_hot, test_size=0.2, random_state=42)
        
    #     # Cast X_train as float for matrix multiplication
    #     X_train = tf.cast(X_train, tf.float64)

    #     # Define the model, optimizer, and original loss
    #     model = tf.keras.models.Sequential([
    #         tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X.shape[1],))])
    #     optimizer = tf.keras.optimizers.Adam()
    #     binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()


    #     # Define the fairness penalty
    #     def fairness_penalty(y_true, y_pred, A, num_groups):
    #         # Initialize a list to store the positive outcome probabilities for each group
    #         group_probs = []

    #         # Compute the positive outcome probabilities for each group
    #         for i in range(num_groups):
    #             group_indices = tf.argmax(A, axis=1)  # find the index of the 1 in each one-hot encoded vector
    #             group_probs.append(tf.reduce_mean(y_pred[tf.logical_and(group_indices==i, y_true==1)]))

    #         # Compute the maximum positive outcome probability
    #         max_prob = tf.reduce_max(group_probs)

    #         # Compute the disparate impacts as the ratios of the positive outcome probabilities to the max probability
    #         DIs = [prob / max_prob for prob in group_probs]

    #         # Compute the fairness penalty as the sum of the squared differences of the disparate impacts from 1 (perfect fairness) (squared to make it differentiable) 
    #         fairness_penalty = tf.reduce_sum([tf.square(DI - 1) for DI in DIs])
            
    #         return fairness_penalty
        
    #     # Define the custom loss function
    #     def custom_loss(y_true, y_pred, A):
    #         lambda_fairness = 0.1
    #         original_loss = binary_cross_entropy(y_true, y_pred)
    #         penalty = fairness_penalty(y_true, y_pred, A, num_groups)
    #         return original_loss + lambda_fairness * penalty
        

    #     # Define the custom training step
    #     @tf.function
    #     def train_step(X, y, A):
    #         with tf.GradientTape() as tape:
    #             predictions = model(X)
    #             loss = custom_loss(y, predictions, A)
    #         grads = tape.gradient(loss, model.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #         return loss
        
    #     # Train the model
    #     num_epochs = 1000
    #     for epoch in range(num_epochs):
    #         train_step(X_train, y_train, A_train)
        

    #     # Make predictions
    #     predictions_test = model(X_test)

    #     # Calculate the accuracy of the model
    #     # model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)
    #     model_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, tf.round(predictions_test)), tf.float32))
    #     # Calculate the AUC score
    #     auc_score = roc_auc_score(y_test, predictions_test.numpy())

    #     # Calculate the fairness metric (demographic parity)
    #     fairness_metric = fairness_penalty(y_test, predictions_test, A_test, num_groups).numpy()

    #     end_time = time.time()  # End measuring execution time
    #     execution_time = end_time - start_time
    #     print("Execution time: {:.2f} seconds".format(execution_time))
    #     print("Predictions Data Type", type(predictions_test.numpy().tolist()))
    #     print("Model Accuracy", type(model_accuracy.numpy()))
    #     print("AUC Score", type(auc_score))

    #     return predictions_test.numpy().tolist(), model_accuracy.numpy().astype(np.float64), auc_score


    # def logistic_regression_equal_opportunity(self):
    #     start_time = time.time()
    #     X = self.df.drop(columns=[self.target_variable, self.sensitive_attribute]).values
    #     y = self.df[self.target_variable].values

    #     # Get unique values for the sensitive attribute and create a mapping
    #     unique_values = self.df[self.sensitive_attribute].unique()
    #     mapping_dict = {value: i for i, value in enumerate(unique_values)}
    #     num_groups = len(unique_values)
    #     print("Number of Groups: ", num_groups)
        
    #     # Apply the mapping to the column and convert to integers
    #     A = self.df[self.sensitive_attribute].map(mapping_dict).values.astype(np.int32)

    #     # One-hot encode the sensitive attribute
    #     one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #     A_one_hot = one_hot_encoder.fit_transform(A.reshape(-1, 1))

    #     # Split the data into training and test sets
    #     X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A_one_hot, test_size=0.2, random_state=42)
        
    #     # Cast X_train as float for matrix multiplication
    #     X_train = tf.cast(X_train, tf.float64)

    #     # Define the model, optimizer, and original loss
    #     model = tf.keras.models.Sequential([
    #         tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X.shape[1],))])
    #     optimizer = tf.keras.optimizers.Adam()
    #     binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()


    #     # Define the fairness penalty
    #     def fairness_penalty(y_true, y_pred, A, num_groups):
    #         # Initialize a list to store the approximated true positive rates for each group
    #         approx_tprs = []

    #         # Compute the approximated true positive rate for each group
    #         for i in range(num_groups):
    #             group_indices = tf.argmax(A, axis=1)  # find the index of the 1 in each one-hot encoded vector
    #             approx_tprs.append(tf.reduce_mean(y_pred[tf.logical_and(group_indices==i, y_true==1)]))

    #         # Compute the maximum absolute difference between any two group's ApproxTPR
    #         max_diff = tf.reduce_max([tf.abs(approx_tprs[i] - approx_tprs[j]) for i in range(num_groups) for j in range(i+1, num_groups)])

    #         return max_diff
        
    #     # Define the custom loss function
    #     def custom_loss(y_true, y_pred, A):
    #         lambda_fairness = 0.1
    #         original_loss = binary_cross_entropy(y_true, y_pred)
    #         penalty = fairness_penalty(y_true, y_pred, A, num_groups)
    #         return original_loss + lambda_fairness * penalty
        

    #     # Define the custom training step
    #     @tf.function
    #     def train_step(X, y, A):
    #         with tf.GradientTape() as tape:
    #             predictions = model(X)
    #             loss = custom_loss(y, predictions, A)
    #         grads = tape.gradient(loss, model.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #         return loss
        
    #     # Train the model
    #     num_epochs = 1000
    #     for epoch in range(num_epochs):
    #         train_step(X_train, y_train, A_train)
        

    #     # Make predictions
    #     predictions_test = model(X_test)

    #     # Calculate the accuracy of the model
    #     # model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)
    #     model_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, tf.round(predictions_test)), tf.float32))
    #     # Calculate the AUC score
    #     auc_score = roc_auc_score(y_test, predictions_test.numpy())

    #     # Calculate the fairness metric (demographic parity)
    #     fairness_metric = fairness_penalty(y_test, predictions_test, A_test, num_groups).numpy()

    #     end_time = time.time()  # End measuring execution time
    #     execution_time = end_time - start_time
    #     print("Execution time: {:.2f} seconds".format(execution_time))
    #     print("Predictions Data Type", type(predictions_test.numpy().tolist()))
    #     print("Model Accuracy", type(model_accuracy.numpy()))
    #     print("AUC Score", type(auc_score))

    #     return predictions_test.numpy().tolist(), model_accuracy.numpy().astype(np.float64), auc_score
