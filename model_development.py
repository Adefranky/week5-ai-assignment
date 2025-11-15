# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Generate a synthetic dataset representing EHR data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(20, 90, 1000),
    'gender': np.random.choice(['Male', 'Female'], 1000),
    'length_of_stay': np.random.randint(1, 15, 1000),
    'previous_admissions': np.random.randint(0, 5, 1000),
    'comorbidity_score': np.random.randint(0, 10, 1000),
    'readmitted': np.random.choice([0, 1], 1000, p=[0.77, 0.23])  # Imbalance based on real-world trends
})

# Separate features and target
X = data.drop('readmitted', axis=1)
y = data['readmitted']

# Define preprocessing for numeric and categorical features
numeric_features = ['age', 'length_of_stay', 'previous_admissions', 'comorbidity_score']
categorical_features = ['gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data into train, test, validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Generate a real confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate precision and recall from test predictions
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\nPrecision Score: {precision:.2f}")
print(f"Recall Score: {recall:.2f}")

# Hypothetical confusion matrix for demonstration
TP = 120
FP = 80
FN = 30
TN = 770

precision_hyp = TP / (TP + FP)
recall_hyp = TP / (TP + FN)

print("\nFrom hypothetical confusion matrix:")
print(f"Precision: {precision_hyp:.2f}")
print(f"Recall: {recall_hyp:.2f}")
