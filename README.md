# Spark Anomaly Bid Detection Framework

**Spark Anomaly Bid Detection Framework** is a big-data-powered project leveraging Apache Spark to detect fraudulent bids in online advertising. This project showcases scalable machine learning pipelines for anomaly detection, handling large-scale datasets effectively. The methodology integrates Spark MLlib with traditional Python-based machine learning libraries to develop and evaluate models capable of identifying fraudulent bids.

## Table of Contents
- [Introduction](#introduction)
- [Context and Challenges](#context-and-challenges)
- [Features](#features)
- [Methodology](#methodology)
  - [Spark Machine Learning Pipeline](#spark-machine-learning-pipeline)
  - [Supervised Learning Models](#supervised-learning-models)
- [Results and Evaluation](#results-and-evaluation)
- [Setup and Usage](#setup-and-usage)
- [Files in the Repository](#files-in-the-repository)
- [Authors](#authors)

## Introduction
This project leverages the distributed computing power of Apache Spark to create scalable pipelines for detecting anomalies in online bid data. By integrating Spark MLlib with conventional machine learning libraries like scikit-learn, we build, train, and evaluate supervised learning models to distinguish between legitimate and fraudulent bids in digital advertising.

## Context and Challenges
Online advertising involves real-time bidding (RTB), where advertisers compete for ad space through auctions. Fraudulent activities such as fake clicks and bots generate misleading bids, resulting in financial losses. Identifying such anomalies is crucial for maintaining ad market integrity. 

Challenges in anomaly bid detection include:
- Handling massive datasets in real-time.
- Feature selection for distinguishing fraudulent behavior.
- Model robustness against evolving fraud patterns.

## Features
- **Big Data Processing**: Apache Spark-based MLlib pipeline enables efficient distributed processing.
- **Supervised Learning**: Utilizes both Spark-based and local machine learning models.
- **Automated Feature Engineering**: Preprocessing and feature selection tailored for bid fraud detection.
- **Scalability**: Handles large-scale datasets efficiently with Spark’s distributed framework.
- **Model Evaluation**: Includes accuracy, precision, recall, and custom fraud detection metrics.

## Methodology

### Spark Machine Learning Pipeline
1. **Data Preprocessing**:
   - Relevant features such as bid ID, bid floor price, location, device type, and connection type are selected.
   - Missing values are imputed based on statistical analysis.
   - Categorical features are encoded using Spark MLlib’s `StringIndexer`.

2. **Model Training**:
   - **Logistic Regression**: A baseline model for classifying bids as fraudulent or legitimate.
   - **Gradient Boosted Trees (GBTClassifier)**: A more robust model that leverages ensemble learning to capture complex fraud patterns.

3. **Evaluation & Predictions**:
   - Validation data is used to evaluate model performance using precision, recall, and a weighted fraud-detection metric.
   - Predictions on test data are saved as CSV for further analysis.

### Supervised Learning Models
1. **Logistic Regression**:
   - Chosen for its simplicity and efficiency in handling large datasets.
   - Provides interpretability in feature importance.

2. **Gradient Boosted Trees (GBTClassifier)**:
   - Offers superior performance due to its ability to capture complex, non-linear fraud patterns.
   - Achieved higher accuracy and precision compared to Logistic Regression.

## Results and Evaluation
- **Logistic Regression**:
  - Accuracy: ~99.5%
  - Precision: 99.99%
  - Recall: 99.49%

- **Gradient Boosted Trees**:
  - Accuracy: ~99.8%
  - Precision: 99.97%
  - Recall: 99.78%
  - **Final Metric**: 99.82% (weighted fraud detection score)

The GBTClassifier model significantly outperformed logistic regression, demonstrating its effectiveness in identifying fraudulent bids. Due to its high precision, it effectively minimizes false positives, reducing financial losses from fraudulent activity.

## Setup and Usage
### Requirements
Install dependencies using:
```bash
pip install pyspark scikit-learn pandas
```

### Running the Code
Run the Jupyter notebook:
```
Anomaly_Detection.ipynb
```

## Files in the Repository
- **`Anomaly_Detection.ipynb`**: Main Jupyter notebook for anomaly bid detection using Spark MLlib.


## Authors
- **Daniel Dahan** (ID: 345123624)
- **Simon Bellilty** (ID: 345233563)

## Acknowledgments
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
