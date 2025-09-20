import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import os

# Paths
DATASET_PATH = 'data/loan_data.csv'
MODEL_PATH = 'models/loan_pipeline.pkl'

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Handle missing values
for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Features and target
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Pipeline: OneHotEncode categoricals + RandomForest
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X, y)

# Create models folder if not exists
os.makedirs('models', exist_ok=True)

# Save pipeline
joblib.dump(pipeline, MODEL_PATH)

print("✅ Model trained and saved successfully!")
