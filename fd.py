import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('Fraud.csv')

# One-hot encode the 'type' column
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Drop 'nameOrig' and 'nameDest' as they are identifiers
data = data.drop(['nameOrig', 'nameDest'], axis=1)

# Handle cases where balances don't match expected patterns
data['errorOrg'] = data['newbalanceOrig'] - (data['oldbalanceOrg'] - data['amount'])
data['errorDest'] = data['newbalanceDest'] - (data['oldbalanceDest'] + data['amount'])

# Features and target
X = data.drop(['isFraud'], axis=1)
y = data['isFraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Set parameters for the LightGBM model with fine-tuning
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',  # Try 'dart' for better accuracy if needed
    'metric': 'binary_logloss',
    'num_leaves': 20,  # Reduced num_leaves to prevent overfitting
    'learning_rate': 0.01,  # Smaller learning rate for better accuracy
    'num_iterations': 1000,  # More iterations to compensate for the small learning rate
    'max_depth': 10,  # Limit depth of trees to avoid overfitting
    'feature_fraction': 0.8,  # Use 80% of features to speed up training
    'bagging_fraction': 0.7,  # Use 70% of data to prevent overfitting
    'bagging_freq': 5,  # Perform bagging every 5 iterations
    'max_bin': 255,  # Increase max_bin for better accuracy
    'lambda_l1': 0.1,  # L1 regularization to control overfitting
    'lambda_l2': 0.1,  # L2 regularization to control overfitting
    'min_data_in_leaf': 50,  # Minimum data in one leaf to reduce overfitting
    'min_sum_hessian_in_leaf': 10,  # Minimum sum of hessian in one leaf
    'num_threads': 4  # Use 4 CPU cores for parallel processing
}

# Train the model
clf = lgb.train(params, train_data, num_boost_round=100)

# Predictions
y_pred = clf.predict(X_test)
y_pred = [1 if x > 0.5 else 0 for x in y_pred]

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
