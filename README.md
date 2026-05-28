# Bank-fraud-detection
# Project 1 - Bank Fraud Detection

## What is this project?
This project is about detecting fraud in bank
transactions using Machine Learning.
I made this project as part of my AI internship program.

## Why I chose this topic
Bank fraud is a big problem today. Many people lose
money because of fraud transactions. I wanted to make
a system that can automatically find out if a
transaction is fraud or not.

## How it works
1. Created transaction data with normal and fraud cases
2. Made graphs to understand the data
3. Trained Logistic Regression and Random Forest models
4. Random Forest gave better accuracy
5. Tested on new transactions to see if it works

## How to run
pip install -r requirements.txt
python fraud_detection.py

## Tools Used
- Python
- pandas
- numpy
- matplotlib
- scikit-learn

## Result
- Random Forest Accuracy : around 97-98%
- Model can correctly identify fraud transactions

## What I learned
- How to handle imbalanced data
- What is Random Forest and Logistic Regression
- How to check model accuracy using confusion matrix