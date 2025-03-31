import numpy as np
import pandas as pd

# Load processed data
df = pd.read_csv("../data/X_train_processed.csv")

# Ensure column names are clean
df.columns = df.columns.str.strip().str.lower()

# Debugging: Print available columns
# print("Columns in dataset:", df.columns.tolist())

# Check if 'lstat' exists before using it
if "lstat" in df.columns:
    df["lstat_sq"] = df["lstat"] ** 2  
else:
    print(" Warning: 'lstat' column not found!")

if "tax" in df.columns and "rm" in df.columns:
    df["tax_to_rm"] = df["tax"] / df["rm"]
else:
    print(" Warning: 'tax' or 'rm' column not found!")

if "crim" in df.columns:
    df["log_crim"] = np.log1p(df["crim"])
else:
    print(" Warning: 'crim' column not found!")

# Save the processed dataset separately
df.to_csv("../data/X_train_with_new_features.csv", index=False)
print(" Data saved successfully!")
