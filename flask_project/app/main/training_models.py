
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

class TrainingModels:
    def __init__(self, df_path, target_variable, sensitive_attribute):
        self.df = pd.read_csv(df_path)
        self.target_variable = target_variable
        self.sensitive_attribute = sensitive_attribute
        
    
    def train_model(self, model_type, fairness_definition):
        if model_type == 'logistic_regression':
            if fairness_definition == 'group_unawareness':
                return self.logistic_regression_group_unawareness()
            elif fairness_definition == 'demographic_parity':
                return self.logistic_regression_demographic_parity()
            elif fairness_definition == 'disparate_impact':
                return self.logistic_regression_disparate_impact()
            elif fairness_definition == 'equal_opportunity':
                return self.logistic_regression_equal_opportunity()
            elif fairness_definition == 'equal_odds':
                return self.logistic_regression_equal_odds()
            elif fairness_definition == 'positive_predictive_value_parity':
                return self.logistic_regression_positive_predictive_value_parity()
            elif fairness_definition == 'false_positive_ratio_parity':
                return self.logistic_regression_false_positive_ratio_parity()
            elif fairness_definition == 'negative_predictive_value_parity':
                return self.logistic_regression_negative_predictive_value_parity()
            else:
                return "Error: Invalid fairness definition."
        elif model_type == 'perceptron':
            if fairness_definition == 'group_unawareness':
                return self.perceptron_group_unawareness()
            elif fairness_definition == 'demographic_parity':
                return self.perceptron_demographic_parity()
            elif fairness_definition == 'disparate_impact':
                return self.perceptron_disparate_impact()
            elif fairness_definition == 'equal_opportunity':
                return self.perceptron_equal_opportunity()
            elif fairness_definition == 'equal_odds':
                return self.perceptron_equal_odds()
            elif fairness_definition == 'positive_predictive_value_parity':
                return self.perceptron_positive_predictive_value_parity()
            elif fairness_definition == 'false_positive_ratio_parity':
                return self.perceptron_false_positive_ratio_parity()
            elif fairness_definition == 'negative_predictive_value_parity':
                return self.perceptron_negative_predictive_value_parity()
            else:
                return "Error: Invalid fairness definition."
        elif model_type == 'neural_network':
            if fairness_definition == 'group_unawareness':
                return self.neural_network_group_unawareness()
            elif fairness_definition == 'demographic_parity':
                return self.neural_network_demographic_parity()
            elif fairness_definition == 'disparate_impact':
                return self.neural_network_disparate_impact()
            elif fairness_definition == 'equal_opportunity':
                return self.neural_network_equal_opportunity()
            elif fairness_definition == 'equal_odds':
                return self.neural_network_equal_odds()
            elif fairness_definition == 'positive_predictive_value_parity':
                return self.neural_network_positive_predictive_value_parity()
            elif fairness_definition == 'false_positive_ratio_parity':
                return self.neural_network_false_positive_ratio_parity()
            elif fairness_definition == 'negative_predictive_value_parity':
                return self.neural_network_negative_predictive_value_parity()
            else:
                return "Error: Invalid fairness definition."
        else:
            return "Error: Invalid model type."
        



    def logistic_regression_demographic_parity(self):
        start_time = time.time()
        # Separate the features and target
        X = self.df.drop(columns=[self.target_variable, self.sensitive_attribute]).values
        y = self.df[self.target_variable].values
        
        # Get unique values for the sensitive attribute and create a mapping
        unique_values = self.df[self.sensitive_attribute].unique()
        mapping_dict = {value: i for i, value in enumerate(unique_values)}
        
        # Apply the mapping to the column and convert to integers
        A = self.df[self.sensitive_attribute].map(mapping_dict).values.astype(np.int32)

        # One-hot encode the sensitive attribute
        one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
        A_one_hot = one_hot_encoder.fit_transform(A.reshape(-1, 1))
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A_one_hot, test_size=0.2, random_state=42)
        
        # Cast X_train as float for matrix multiplication
        X_train = tf.cast(X_train, tf.float64)

        # Initialize the weights and bias
        w = tf.Variable(tf.random.normal([X.shape[1], 1], dtype = tf.float64))
        b = tf.Variable(tf.zeros([1], dtype = tf.float64))

        # Define the logistic regression model
        def model(X):
            return tf.math.sigmoid(tf.matmul(X, w) + b)

        # Define the fairness penalty
        def fairness_penalty(A_one_hot, predictions):
            group_predictions = tf.matmul(tf.transpose(A_one_hot), predictions)
            group_counts = tf.reduce_sum(A_one_hot, axis=0)
            group_averages = group_predictions / group_counts[:, tf.newaxis]
            max_diff = tf.reduce_max(group_averages) - tf.reduce_min(group_averages)
            return max_diff

            # group_0_mask = tf.equal(A, 0)
            # group_1_mask = tf.equal(A, 1)
            # predictions_0 = tf.boolean_mask(predictions, group_0_mask)
            # predictions_1 = tf.boolean_mask(predictions, group_1_mask)
            # return tf.abs(tf.reduce_mean(predictions_0) - tf.reduce_mean(predictions_1))

        # Define the custom loss function
        def loss_fn(y_true, y_pred, A_one_hot, lambda_fairness):
            log_loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
            fairness_loss = fairness_penalty(A_one_hot, y_pred)
            return log_loss + lambda_fairness * fairness_loss

        # Define the optimizer
        optimizer = tf.optimizers.Adam()

        # Train the model
        for _ in range(1000):  # number of training iterations
            with tf.GradientTape() as tape:
                X_train_tf = tf.cast(X_train, tf.float64)
                predictions = model(X_train_tf)
                # predictions = model(X_train)
                loss = loss_fn(y_train, predictions, A_train, lambda_fairness=1.0)  # adjust lambda_fairness as needed
            grads = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(zip(grads, [w, b]))

        # Make predictions
        predictions_test = model(X_test)

        # Calculate the accuracy of the model
        model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)

        # Calculate the AUC score
        auc_score = roc_auc_score(y_test, predictions_test.numpy())

        # Calculate the fairness metric (demographic parity)
        fairness_metric = fairness_penalty(A_test, predictions_test).numpy()

        end_time = time.time()  # End measuring execution time
        execution_time = end_time - start_time
        print("Execution time: {:.2f} seconds".format(execution_time))

        return predictions_test.numpy().tolist(), model_accuracy, auc_score

    def logistic_regression_disparate_impact(self):
        start_time = time.time()
        X = self.df.drop(columns=[self.target_variable, self.sensitive_attribute]).values
        y = self.df[self.target_variable].values

        # Get unique values for the sensitive attribute and create a mapping
        unique_values = self.df[self.sensitive_attribute].unique()
        mapping_dict = {value: i for i, value in enumerate(unique_values)}
        num_groups = len(unique_values)
        print("Number of Groups: ", num_groups)
        
        # Apply the mapping to the column and convert to integers
        A = self.df[self.sensitive_attribute].map(mapping_dict).values.astype(np.int32)

        # One-hot encode the sensitive attribute
        one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
        A_one_hot = one_hot_encoder.fit_transform(A.reshape(-1, 1))

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A_one_hot, test_size=0.2, random_state=42)
        
        # Cast X_train as float for matrix multiplication
        X_train = tf.cast(X_train, tf.float64)

        # Define the model, optimizer, and original loss
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X.shape[1],))])
        optimizer = tf.keras.optimizers.Adam()
        binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()


        # Define the fairness penalty
        def fairness_penalty(y_true, y_pred, A, num_groups):
            # Initialize a list to store the positive outcome probabilities for each group
            group_probs = []

            # Compute the positive outcome probabilities for each group
            for i in range(num_groups):
                group_indices = tf.argmax(A, axis=1)  # find the index of the 1 in each one-hot encoded vector
                group_probs.append(tf.reduce_mean(y_pred[tf.logical_and(group_indices==i, y_true==1)]))

            # Compute the maximum positive outcome probability
            max_prob = tf.reduce_max(group_probs)

            # Compute the disparate impacts as the ratios of the positive outcome probabilities to the max probability
            DIs = [prob / max_prob for prob in group_probs]

            # Compute the fairness penalty as the sum of the squared differences of the disparate impacts from 1 (perfect fairness) (squared to make it differentiable) 
            fairness_penalty = tf.reduce_sum([tf.square(DI - 1) for DI in DIs])
            
            return fairness_penalty
        
        # Define the custom loss function
        def custom_loss(y_true, y_pred, A):
            lambda_fairness = 0.1
            original_loss = binary_cross_entropy(y_true, y_pred)
            penalty = fairness_penalty(y_true, y_pred, A, num_groups)
            return original_loss + lambda_fairness * penalty
        

        # Define the custom training step
        @tf.function
        def train_step(X, y, A):
            with tf.GradientTape() as tape:
                predictions = model(X)
                loss = custom_loss(y, predictions, A)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss
        
        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            train_step(X_train, y_train, A_train)
        

        # Make predictions
        predictions_test = model(X_test)

        # Calculate the accuracy of the model
        # model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)
        model_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, tf.round(predictions_test)), tf.float32))
        # Calculate the AUC score
        auc_score = roc_auc_score(y_test, predictions_test.numpy())

        # Calculate the fairness metric (demographic parity)
        fairness_metric = fairness_penalty(y_test, predictions_test, A_test, num_groups).numpy()

        end_time = time.time()  # End measuring execution time
        execution_time = end_time - start_time
        print("Execution time: {:.2f} seconds".format(execution_time))
        print("Predictions Data Type", type(predictions_test.numpy().tolist()))
        print("Model Accuracy", type(model_accuracy.numpy()))
        print("AUC Score", type(auc_score))

        return predictions_test.numpy().tolist(), model_accuracy.numpy().astype(np.float64), auc_score
