import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.compose import ColumnTransformer

def preprocess_data(df_path, target_variable, sensitive_attribute):
    df = pd.read_csv(df_path)
    # Remove all NaN values
    df = df.dropna()
    
    # Separate numerical and categorical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object','string']).columns


    # Get sets of unique counts for each type of column
    num_unique_counts = set(df[numerical_cols].nunique())
    cat_unique_counts = set(df[categorical_cols].nunique())

    # Ensure target variable and sensitive attribute are not included in the numerical and categorical columns
    numerical_cols = numerical_cols.drop(target_variable) if target_variable in numerical_cols else numerical_cols
    numerical_cols = numerical_cols.drop(sensitive_attribute) if sensitive_attribute in numerical_cols else numerical_cols
    categorical_cols_preprocessing = categorical_cols.drop([target_variable, sensitive_attribute])

    # Identify numerical columns to drop based on matching unique counts
    cols_to_drop = [col for col in numerical_cols if df[col].nunique() in cat_unique_counts]


    # Update numerical columns by dropping identified columns
    numerical_cols = numerical_cols.drop(cols_to_drop)
    # print("numerical_cols:", numerical_cols)

    """ Note: Removing standard scaling for now if you do this for prejudice remover, you'll have to add standard scaling to that file"""
    preprocessor = ColumnTransformer(
        transformers=[
            # ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols_preprocessing)])
    
    le = LabelEncoder()
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype(str)


    # Separate the features and target

    X = preprocessor.fit_transform(df.drop(columns=[target_variable, sensitive_attribute]))
    y = le.fit_transform(df[target_variable]).astype(np.float32)    

    
    # Get unique values for the sensitive attribute and create a mapping
    unique_values = df[sensitive_attribute].unique()

    mapping_dict = {value: i for i, value in enumerate(unique_values)}

    
    
    # Apply the mapping to the column and convert to integers
    A = df[sensitive_attribute].map(mapping_dict).values.astype(np.int32)

    # One-hot encode the sensitive attribute
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    A_one_hot = one_hot_encoder.fit_transform(A.reshape(-1, 1))
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A_one_hot, test_size=0.2, random_state=42)

    # Convert the one hot encoded values back to their integer representation
    A_train_int = one_hot_encoder.inverse_transform(A_train).flatten()
    A_test_int = one_hot_encoder.inverse_transform(A_test).flatten()

    # Convert the integer representation back to the original values using the inverse of the mapping dictionary
    inverse_mapping_dict = {v: k for k, v in mapping_dict.items()}
    A_train_original = np.array([inverse_mapping_dict[val] for val in A_train_int])
    A_test_original = np.array([inverse_mapping_dict[val] for val in A_test_int])

    return X_train, y_train, A_train, X_test, y_test, A_test, A_train_original, A_test_original