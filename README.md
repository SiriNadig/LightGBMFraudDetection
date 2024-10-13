# Fraud Detection Using LightGBM

This project focuses on detecting fraudulent transactions from a large dataset containing millions of entries. The model is built using LightGBM, a highly efficient gradient boosting algorithm, designed for fast processing of large datasets. It identifies patterns and key features that help in detecting fraudulent activities.

## Project Overview

- **Objective**: Develop a robust fraud detection model for large-scale financial transactions using LightGBM.
- **Key Features**:
  - Efficient handling of large datasets with millions of rows.
  - One-hot encoding of categorical features.
  - Incorporation of error metrics (`errorOrg`, `errorDest`) to enhance predictive accuracy.
  - Model performance evaluation using accuracy score, confusion matrix, and classification report.
  
## Model Performance

The model achieves high accuracy in identifying fraudulent transactions. It was evaluated using a hold-out test set, and the results were measured using standard classification metrics.

## Instructions

1. Clone the repository.
2. Install necessary dependencies via `requirements.txt`.
3. Download the dataset from the link below and place it in the working directory.

[**Link to Dataset**](https://www.kaggle.com/datasets/vardhansiramdasu/fraudulent-transactions-prediction/data)

4. Run the `fd.py` script to train the model and evaluate the results.
