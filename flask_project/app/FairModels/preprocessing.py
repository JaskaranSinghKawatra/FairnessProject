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

    # print("numerical_cols:", numerical_cols)
    # print("categorical_cols:", categorical_cols)

    # Get sets of unique counts for each type of column
    num_unique_counts = set(df[numerical_cols].nunique())
    cat_unique_counts = set(df[categorical_cols].nunique())

    # print("num_unique_counts:", num_unique_counts)
    # print("cat_unique_counts:", cat_unique_counts)

    # Ensure target variable and sensitive attribute are not included in the numerical and categorical columns
    numerical_cols = numerical_cols.drop(target_variable) if target_variable in numerical_cols else numerical_cols
    numerical_cols = numerical_cols.drop(sensitive_attribute) if sensitive_attribute in numerical_cols else numerical_cols
    categorical_cols_preprocessing = categorical_cols.drop([target_variable, sensitive_attribute])

    # print("numerical_cols:", numerical_cols)
    # print("categorical_cols_preprocessing:", categorical_cols_preprocessing)
    
    # Identify numerical columns to drop based on matching unique counts
    cols_to_drop = [col for col in numerical_cols if df[col].nunique() in cat_unique_counts]

    # print("cols_to_drop:", cols_to_drop)

    # Update numerical columns by dropping identified columns
    numerical_cols = numerical_cols.drop(cols_to_drop)
    # print("numerical_cols:", numerical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols_preprocessing)])
    
    le = LabelEncoder()
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype(str)

    # print("df columns:", df.columns)
    # print("df numerical columns:", df[numerical_cols].columns)
    # print("df categorical columns:", df[categorical_cols_preprocessing].columns)

    # Separate the features and target

    X = preprocessor.fit_transform(df.drop(columns=[target_variable, sensitive_attribute]))
    y = le.fit_transform(df[target_variable]).astype(np.float32)    

    # X = df.drop(columns=[target_variable, sensitive_attribute]).values.astype(np.float32)
    # y = df[target_variable].values.astype(np.float32)
    
    # Get unique values for the sensitive attribute and create a mapping
    unique_values = df[sensitive_attribute].unique()
    # print("Unique Values in DataFrame: ", unique_values)

    mapping_dict = {value: i for i, value in enumerate(unique_values)}
    # print("Mapping Dictionary: ", mapping_dict)

    
    
    # Apply the mapping to the column and convert to integers
    A = df[sensitive_attribute].map(mapping_dict).values.astype(np.int32)

    # One-hot encode the sensitive attribute
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    A_one_hot = one_hot_encoder.fit_transform(A.reshape(-1, 1))
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A_one_hot, test_size=0.2, random_state=42)

    return X_train, y_train, A_train, X_test, y_test, A_test