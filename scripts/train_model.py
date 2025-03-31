import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Loading dataset...")
df = pd.read_csv("../data/BostonHousing.csv")

# Handle missing values
df.fillna(df.median(), inplace=True)

# Identify categorical and numerical features
categorical_features = ['chas']
numerical_features = df.columns.difference(['medv', 'chas'])

logging.info("Defining preprocessing pipeline...")
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Split dataset into features and target
X = df.drop(columns=['medv'])
y = df['medv']

# Split the data
logging.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformations
logging.info("Applying transformations...")
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train the model
logging.info("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Perform Cross-Validation (optional hyperparameter tuning step)
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
logging.info(f"Cross-validation R² scores: {cv_scores}")
logging.info(f"Mean R² score: {cv_scores.mean():.4f}")

# Save model and preprocessor
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

logging.info("Saving model and preprocessor...")
joblib.dump(model, os.path.join(model_dir, "linear_regression_model.pkl"))
joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))

logging.info("Model training completed and saved.")
