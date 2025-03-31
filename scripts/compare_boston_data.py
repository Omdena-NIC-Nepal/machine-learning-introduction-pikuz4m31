import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define folders
data_folder = "../data"
comparison_folder = "../reports"
compare_images_folder = "../compare_images"

# Ensure comparison folders exist
os.makedirs(comparison_folder, exist_ok=True)
os.makedirs(compare_images_folder, exist_ok=True)

# Load original dataset
boston_file = os.path.join(data_folder, "BostonHousing.csv")
if not os.path.exists(boston_file):
    raise FileNotFoundError("BostonHousing.csv not found in data folder!")

boston_df = pd.read_csv(boston_file)

# Datasets to compare
datasets_to_compare = [
    "X_train_processed.csv",
    "X_test_processed.csv",
    "X_train_with_new_features.csv"
]

def compare_datasets(original, modified, name):
    """
    Compare the original dataset with a modified dataset.
    Detects column changes and numerical differences.
    """
    comparison_results = {
        "Dataset": name,
        "Status": "Comparison Completed",
        "Added_Columns": 0,
        "Removed_Columns": 0,
        "Total_Numerical_Differences": 0
    }

    # Identify added and removed columns
    original_columns = set(original.columns)
    modified_columns = set(modified.columns)

    added_columns = list(modified_columns - original_columns)
    removed_columns = list(original_columns - modified_columns)

    comparison_results["Added_Columns"] = len(added_columns)
    comparison_results["Removed_Columns"] = len(removed_columns)

    # Save column changes
    column_changes_df = pd.DataFrame({
        "Added_Columns": [", ".join(added_columns) if added_columns else "None"],
        "Removed_Columns": [", ".join(removed_columns) if removed_columns else "None"]
    })
    column_changes_df.to_csv(os.path.join(comparison_folder, f"{name}_column_changes.csv"), index=False)

    # Handle missing columns in numerical comparison
    common_columns = original.select_dtypes(include="number").columns.intersection(
        modified.select_dtypes(include="number").columns
    )

    num_diff = modified[common_columns].describe() - original[common_columns].describe()

    # Count non-zero differences
    comparison_results["Total_Numerical_Differences"] = (num_diff.abs() > 0).sum().sum()

    # Save numerical differences
    num_diff.to_csv(os.path.join(comparison_folder, f"{name}_numerical_diff.csv"))

    # Visualization - Column Changes
    plt.figure(figsize=(8, 5))
    plt.bar(["Added Columns", "Removed Columns"], [len(added_columns), len(removed_columns)], color=['green', 'red'])
    plt.xlabel("Change Type")
    plt.ylabel("Number of Columns")
    plt.title(f"Column Changes in {name}")
    plt.savefig(os.path.join(compare_images_folder, f"{name}_column_changes.png"))
    plt.close()

    # Heatmap for numerical changes
    plt.figure(figsize=(10, 6))
    sns.heatmap(num_diff, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
    plt.title(f"Numerical Changes in {name}")
    plt.savefig(os.path.join(compare_images_folder, f"{name}_numerical_changes.png"))
    plt.close()

    return comparison_results

# Compare datasets and generate summary
summary_list = []
for dataset in datasets_to_compare:
    dataset_path = os.path.join(data_folder, dataset)
    if not os.path.exists(dataset_path):
        print(f"Skipping {dataset}: File not found!")
        summary_list.append({"Dataset": dataset, "Status": "File Not Found", "Added_Columns": "N/A", "Removed_Columns": "N/A", "Total_Numerical_Differences": "N/A"})
        continue

    try:
        modified_df = pd.read_csv(dataset_path)
        result = compare_datasets(boston_df, modified_df, dataset.replace(".csv", ""))
        summary_list.append(result)
        print(f"Comparison completed for {dataset}")
    except Exception as e:
        summary_list.append({"Dataset": dataset, "Status": f"Error: {e}", "Added_Columns": "N/A", "Removed_Columns": "N/A", "Total_Numerical_Differences": "N/A"})
        print(f"Error processing {dataset}: {e}")

# Save enhanced summary
summary_df = pd.DataFrame(summary_list)
summary_file = os.path.join(comparison_folder, "comparison_summary.csv")
summary_df.to_csv(summary_file, index=False)

print(f"\nComparison Summary saved to {summary_file}")
