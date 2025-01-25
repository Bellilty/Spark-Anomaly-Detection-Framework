# Spark Anomaly Detection Framework

**Spark Anomaly Detection Framework** is a big-data-powered project leveraging Apache Spark to detect anomalies in datasets with supervised learning techniques. The project showcases scalable machine learning pipelines for anomaly detection, handling large-scale datasets effectively.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
  - [Spark Machine Learning Pipeline](#spark-machine-learning-pipeline)
  - [Standard Machine Learning Models](#standard-machine-learning-models)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Files in the Repository](#files-in-the-repository)
- [Authors](#authors)

## Introduction
This project leverages the distributed computing power of Apache Spark to create scalable pipelines for anomaly detection. It combines Spark MLlib with traditional Python-based libraries like scikit-learn to build, train, and evaluate supervised learning models on labeled datasets.

## Features
- Apache Spark-based MLlib pipeline for distributed machine learning.
- Implementation of both distributed and local supervised learning models.
- Automated feature engineering and preprocessing.
- Anomaly predictions exported to CSV for evaluation.
- Comprehensive model performance metrics, including precision, recall, and F1-score.

## Methodology

### Spark Machine Learning Pipeline
1. **Data Ingestion and Preprocessing**:
   - Data is loaded into Spark DataFrames from various sources.
   - Features are engineered and preprocessed for machine learning tasks.
2. **Model Training**:
   - Spark MLlib models (e.g., Decision Tree, Logistic Regression) are trained in a distributed environment.
   - Models are optimized using hyperparameter tuning and cross-validation.
3. **Evaluation and Prediction**:
   - Models are evaluated using distributed metrics, and predictions are saved to CSV files (`spark_predictions.csv`).

### Standard Machine Learning Models
1. **Model Implementation**:
   - Scikit-learn models such as Random Forest, Gradient Boosting, and SVM are trained on local datasets.
2. **Performance Evaluation**:
   - Metrics such as precision, recall, and F1-score are calculated locally for comparison.
3. **Prediction Output**:
   - Anomaly predictions are exported to a CSV file (`local_predictions.csv`).

## Results
The project effectively detected anomalies with the following highlights:
- **Distributed Processing**: Spark significantly improved scalability for large datasets.
- **Performance Metrics**: High precision and F1-scores indicate the models' reliability in identifying anomalies.

## Setup and Usage
### Requirements
Install the required dependencies:
```bash
pip install pyspark scikit-learn pandas
```

### Running the Code
  Execute ipynb file:
   ```
     Anomaly_Detection.ipynb
   ```

## Files in the Repository
- **`Anomaly_Detection.ipynb`**: Jupyter notebook for distributed anomaly detection and standard supervised learning models.
- **`spark_predictions.csv`**: Output predictions from the Spark pipeline after running the code.
- **`local_predictions.csv`**: Output predictions from local models after running the code.

## Authors
- **Daniel Dahan** (ID: 345123624)
- **Simon Bellilty** (ID: 345233563)

## Acknowledgments
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

