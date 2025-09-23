# CapstoneProject-LowResourceFraudDetection

# Low-Resource Fraud Detection

## Project Overview

This capstone project addresses the critical challenge of digital fraud in **low-resource areas** with unreliable power and internet. These regions often generate limited financial data, which makes it difficult to train traditional machine learning models to accurately detect fraud.

The objective of this project is to develop a robust machine learning model that can effectively identify fraudulent credit card transactions even with a small, imbalanced dataset. The solution prioritizes **recall** to ensure a high rate of fraudulent transactions are caught, thereby minimizing financial losses.

## Key Features

  * **Data Handling**: Utilizes a publicly available, anonymized dataset from Kaggle to simulate a low-resource data environment.
  * **Model Optimization**: Compares and optimizes **Logistic Regression** and **Random Forest Classifier** models.
  * **Imbalance Handling**: Employs **SMOTE (Synthetic Minority Over-sampling Technique)** to address the severe class imbalance in the dataset.
  * **Explainability**: Provides model explainability through **SHAP (SHapley Additive exPlanations)** to help users understand why a specific transaction was flagged as fraudulent.
  * **Interactive App**: An interactive web application built with **Streamlit** allows for real-time transaction testing and visualization of model performance.

## Methodology

The project follows a comprehensive machine learning pipeline:

1.  **Data Acquisition**: The dataset is automatically downloaded from KaggleHub.
2.  **Data Preprocessing**: Duplicate rows are removed, and the dataset is prepared for modeling.
3.  **Model Training**: The data is split, and the minority class is oversampled using SMOTE to balance the training data.
4.  **Model Evaluation**: Models are evaluated using standard metrics (Precision, Recall, F1-Score) and a custom scorer that prioritizes recall.
5.  **Hyperparameter Tuning**: **GridSearchCV** with **StratifiedKFold** is used to find the best hyperparameters for each model.
6.  **Final Prediction & Analysis**: The best-performing model is deployed in a Streamlit app, where its performance metrics are displayed, and individual predictions can be tested and explained.

## Results

The **optimized Logistic Regression model** emerged as the most effective solution for this problem. It demonstrated superior performance in capturing fraudulent transactions due to the custom scoring metric.

  * **Optimized Logistic Regression**:
      * **F1-Score**: 0.8173
      * **Precision**: 0.7303
      * **Recall**: 0.9255

## How to Run the App

To run the interactive fraud detection application locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/low-resource-fraud-detection.git
    cd low-resource-fraud-detection
    ```
2.  **Install dependencies:**
    Ensure you have `pip` and a compatible Python version (3.7+) installed.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the notebook version on Google Colab:**
    ```
    (https://colab.research.google.com/drive/1uNlV1j4PUQuH61NBFPUh6JLp1nNQuU3L?usp=sharing)
    ```
    
 4. **Run the Streamlit application:**
    ```bash
    streamlit run fraud_detection_hf_rag_app.py
    ```
    This will launch the app in your default web browser.


   
    

   

## Future Work

  * Explore other advanced models, such as Gradient Boosting or Neural Networks, on larger, more diverse datasets.
  * Implement real-time data streaming capabilities to simulate a live fraud detection environment.
  * Develop a full MLOps pipeline for continuous model retraining and deployment.
