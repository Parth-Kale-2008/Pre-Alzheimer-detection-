import pandas as pd
import numpy as np 
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df = pd.read_csv("alzheimers_cleaned.csv")
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

X = df.drop(["Diagnosis", "PatientID", "DoctorInCharge"], axis=1)
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))