"""
PART 2 — DATA STRATEGY & PREPROCESSING PIPELINE
Hospital Readmission Risk Prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



np.random.seed(42)
data = pd.DataFrame({
    "age": np.random.randint(18, 95, 1000),
    "gender": np.random.choice(["Male", "Female"], 1000),
    "length_of_stay": np.random.randint(1, 20, 1000),
    "previous_admissions": np.random.randint(0, 6, 1000),
    "comorbidity_score": np.random.randint(0, 15, 1000),
    "discharge_status": np.random.choice(["Home", "Rehab", "Transferred"], 1000),
    "readmitted": np.random.choice([0, 1], 1000, p=[0.75, 0.25])  # target
})




# Create engineered features:
def add_feature_engineering(df):

    # Age binning
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 60, 100],
        labels=["Young", "Middle-aged", "Senior"]
    )

    # Avoid division by zero
    df["admission_ratio"] = df.apply(
        lambda row: row["length_of_stay"] / (row["previous_admissions"] + 1),
        axis=1
    )

    return df


# Apply feature engineering
data = add_feature_engineering(data)

# Separate features and labels
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# Identify numeric and categorical columns
numeric_features = ["age", "length_of_stay", "previous_admissions",
                    "comorbidity_score", "admission_ratio"]

categorical_features = ["gender", "discharge_status", "age_group"]

# Preprocessor (Scaling + OneHot Encoding)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Create final preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])

# Test preprocessing pipeline
X_transformed = preprocessing_pipeline.fit_transform(X)

print("Preprocessing Pipeline Completed Successfully!")
print("Transformed Shape:", X_transformed.shape)

# (Optional) Show first 5 rows before preprocessing
print("\nSample Raw Data:")
print(X.head())
