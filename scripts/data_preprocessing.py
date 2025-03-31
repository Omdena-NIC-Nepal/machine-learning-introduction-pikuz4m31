import os
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
logging.info("Loading dataset...")
df = pd.read_csv("../data/BostonHousing.csv")

# Handle missing values
logging.info("Handling missing values...")
df.fillna(df.median(), inplace=True)

# Identify categorical and numerical features
categorical_features = ['chas']
numerical_features = df.columns.difference(['medv', 'chas'])

# Define preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Split dataset into features and target
X = df.drop(columns=['medv'])
y = df['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformations
logging.info("Applying transformations...")
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

#  Retrieve correct feature names
num_feature_names = numerical_features.tolist()
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
all_feature_names = num_feature_names + cat_feature_names

# Convert to DataFrame with correct column names
X_train_transformed = pd.DataFrame(X_train_transformed, columns=all_feature_names)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=all_feature_names)

# Save preprocessor
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))

# Save processed data
logging.info("Saving preprocessed data...")
X_train_transformed.to_csv("../data/X_train_processed.csv", index=False)
X_test_transformed.to_csv("../data/X_test_processed.csv", index=False)
pd.DataFrame(y_train).to_csv("../data/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("../data/y_test.csv", index=False)

logging.info("Data preprocessing completed.")
