# Churn-Prediction

# Project Structure for Churn Prediction: ETL Pipeline with Spark, Training, and Deployment using Streamlit

```
churn_prediction_pipeline/
├── data/
│   └── raw_data.csv                  # Raw input data (can be sourced from database, etc.)
│   └── transformed_data.csv          # Cleaned and transformed data
├── etl/
│   └── etl_pipeline.py              # ETL pipeline script using PySpark
│   └── transform.py                # Transformations and data cleaning functions
├── src/
│   ├── train.py                    # Model training and evaluation
│   └── model.py                    # Modular model definitions (CatBoost or other)
├── utils/
│   └── helpers.py                  # Helper functions for data loading, preprocessing, and metrics
├── deployment/
│   └── streamlit_app.py            # Streamlit application for deploying the model
├── models/
│   └── churn_model.pkl             # Serialized model saved after training
├── logs/
│   └── etl.log                     # Log file for ETL pipeline runs
├── requirements.txt                   # List of Python dependencies
├── README.md                          # Project overview and setup instructions
├── config.yaml                        # Configuration file for managing settings (paths, parameters, etc.)
└── notebooks/
    └── eda.ipynb                    # Jupyter Notebook for Exploratory Data Analysis
```

### Directory Breakdown:

1. **`data/`**: 
   - Contains raw data (`raw_data.csv`) and transformed data after the ETL step (`transformed_data.csv`).

2. **`etl/`**:
   - **`etl_pipeline.py`**: Contains the entire ETL pipeline implementation using PySpark. It reads raw data, processes it, and stores the transformed data.
   - **`transform.py`**: Contains reusable transformation functions for data cleaning, feature engineering, etc., utilized by `etl_pipeline.py`.

3. **`src/`**:
   - **`train.py`**: The script for training the model. It reads transformed data, splits it into training and test sets, trains the model, evaluates it, and saves it.
   - **`model.py`**: Contains modularized model definitions (e.g., for different machine learning algorithms like CatBoost, Random Forest, etc.). This separation makes the codebase more maintainable.

4. **`utils/`**:
   - **`helpers.py`**: Contains helper functions used across the project, such as data loading, performance metrics calculations, and other utilities.

5. **`deployment/`**:
   - **`streamlit_app.py`**: A Streamlit script to deploy the model. It contains code to load the serialized model and provide an interface for predicting customer churn.

6. **`models/`**:
   - Stores the serialized model file (`churn_model.pkl`) for deployment purposes.

7. **`logs/`**:
   - **`etl.log`**: A log file generated during the ETL process, containing information on transformations and data issues.

8. **`requirements.txt`**:
   - Lists all Python dependencies (e.g., `pyspark`, `catboost`, `streamlit`, `pandas`). This file makes it easy to set up a virtual environment.

9. **`README.md`**:
   - Provides an overview of the project, setup instructions, and a guide for running the ETL pipeline, model training, and deployment.

10. **`config.yaml`**:
    - A YAML file for configuration settings like data paths, training hyperparameters, etc. This allows easy adjustment of project settings.

11. **`notebooks/`**:
    - **`eda.ipynb`**: A Jupyter Notebook to perform Exploratory Data Analysis on the raw data. It helps in understanding the data better before running the ETL pipeline.

### Sample Code for Each Module

#### **`etl/etl_pipeline.py`** (ETL Pipeline Using PySpark)
```python
from pyspark.sql import SparkSession
from etl.transform import clean_data

# Initialize Spark session
spark = SparkSession.builder.appName("ChurnETL").getOrCreate()

# Read raw data
df = spark.read.csv("../data/raw_data.csv", header=True, inferSchema=True)

# Transform data using the cleaning functions
df_clean = clean_data(df)

# Write the transformed data to CSV
df_clean.write.csv("../data/transformed_data.csv", header=True)
```

#### **`src/train.py`** (Model Training)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow.catboost
from catboost import CatBoostClassifier
import joblib
from utils.helpers import load_data

# Load transformed data
data = load_data("../data/transformed_data.csv")
X = data.drop('churn', axis=1)
y = data['churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=10, verbose=100)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Save the model
joblib.dump(model, '../models/churn_model.pkl')
```

#### **`deployment/streamlit_app.py`** (Streamlit App for Deployment)
```python
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('../models/churn_model.pkl')

# Streamlit app
st.title('Customer Churn Prediction')

# User input for features
def user_input_features():
    feature1 = st.number_input('Feature 1 value')
    feature2 = st.number_input('Feature 2 value')
    # Add more features as per dataset
    data = {'feature1': feature1, 'feature2': feature2}
    return pd.DataFrame(data, index=[0])

# Predict churn
input_df = user_input_features()
if st.button('Predict'): 
    prediction = model.predict(input_df)
    if prediction == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")
```

### Summary
This project structure allows you to maintain a clean separation between the different phases of the machine learning workflow (ETL, Training, and Deployment). Each script is modular, making it easy to extend and maintain. You can add or modify the modules independently without affecting the rest of the project.
