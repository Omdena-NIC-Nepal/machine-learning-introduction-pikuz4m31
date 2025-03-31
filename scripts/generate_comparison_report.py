import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Define directories
data_folder = "../data"
comparison_folder = "../reports"
compare_images_folder = "../compare_images"

# Ensure directories exist
os.makedirs(comparison_folder, exist_ok=True)
os.makedirs(compare_images_folder, exist_ok=True)

# Load original dataset (BostonHousing.csv)
boston_df = pd.read_csv(os.path.join(data_folder, "BostonHousing.csv"))

# List of datasets to compare
datasets_to_compare = [
    "X_train_processed.csv",
    "X_test_processed.csv",
    "X_train_with_new_features.csv"
]

# Function to compare datasets
def compare_datasets(original, modified, name):
    comparison_results = {}

    # Identify added and removed columns
    original_columns = set(original.columns)
    modified_columns = set(modified.columns)

    added_columns = list(modified_columns - original_columns)
    removed_columns = list(original_columns - modified_columns)

    comparison_results['Added Columns'] = added_columns
    comparison_results['Removed Columns'] = removed_columns

    # Save column differences
    pd.DataFrame.from_dict(comparison_results, orient='index').to_csv(
        os.path.join(comparison_folder, f"{name}_column_changes.csv")
    )

    # Compute numerical differences
    common_columns = original.columns.intersection(modified.columns)
    num_diff = modified[common_columns].describe() - original[common_columns].describe()
    
    # Save numerical differences
    num_diff.to_csv(os.path.join(comparison_folder, f"{name}_numerical_diff.csv"))

    # Generate column changes plot
    plt.figure(figsize=(10, 5))
    plt.bar(["Added Columns", "Removed Columns"], [len(added_columns), len(removed_columns)], color=['green', 'red'])
    plt.xlabel("Change Type")
    plt.ylabel("Number of Columns")
    plt.title(f"Column Changes in {name}")
    plt.savefig(os.path.join(compare_images_folder, f"{name}_column_changes.png"))
    plt.close()

    # Generate heatmap for numerical differences
    plt.figure(figsize=(12, 6))
    sns.heatmap(num_diff, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(f"Numerical Changes in {name}")
    plt.savefig(os.path.join(compare_images_folder, f"{name}_numerical_changes.png"))
    plt.close()

# Compare all datasets and store results
summary_list = []
for dataset in datasets_to_compare:
    try:
        modified_df = pd.read_csv(os.path.join(data_folder, dataset))
        compare_datasets(boston_df, modified_df, dataset.replace(".csv", ""))
        summary_list.append({"Dataset": dataset, "Status": "Comparison Completed"})
        print(f"Comparison completed for {dataset}")
    except Exception as e:
        summary_list.append({"Dataset": dataset, "Status": f"Error: {e}"})
        print(f"Error processing {dataset}: {e}")

# Save summary
summary_df = pd.DataFrame(summary_list)
summary_csv_path = os.path.join(comparison_folder, "comparison_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

### PDF Report Generation ###
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", style="B", size=16)
        self.cell(200, 10, "Dataset Comparison Report", ln=True, align='C')
        self.ln(10)
    
    def chapter_title(self, title):
        self.set_font("Arial", style="B", size=12)
        self.cell(0, 10, title, ln=True, align='L')
        self.ln(5)
    
    def chapter_body(self, body):
        self.set_font("Arial", size=10)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, caption):
        self.chapter_title(caption)
        self.image(image_path, x=10, w=180)
        self.ln(10)

# Initialize PDF
pdf = PDFReport()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Add Summary Section
pdf.chapter_title("Comparison Summary")
if not summary_df.empty:
    for _, row in summary_df.iterrows():
        pdf.chapter_body(f"{row.iloc[0]}: {row.iloc[1]}")
else:
    pdf.chapter_body("No summary data available.")

# Add Column Change Details
pdf.chapter_title("Column Changes")
for dataset in datasets_to_compare:
    column_changes_path = os.path.join(comparison_folder, f"{dataset.replace('.csv', '')}_column_changes.csv")
    if os.path.exists(column_changes_path):
        changes_df = pd.read_csv(column_changes_path)
        for _, row in changes_df.iterrows():
            pdf.chapter_body(f"{row.iloc[0]}: {row.iloc[1:].to_string(index=False)}")

# Add Numerical Change Details
pdf.chapter_title("Numerical Differences")
for dataset in datasets_to_compare:
    numerical_diff_path = os.path.join(comparison_folder, f"{dataset.replace('.csv', '')}_numerical_diff.csv")
    if os.path.exists(numerical_diff_path):
        num_diff_df = pd.read_csv(numerical_diff_path)
        for _, row in num_diff_df.iterrows():
            pdf.chapter_body(f"{row.iloc[0]}: {row.iloc[1:].to_string(index=False)}")

# Add Visual Comparisons
pdf.chapter_title("Visual Comparisons")
for img_file in sorted(os.listdir(compare_images_folder)):
    if img_file.endswith(".png"):
        pdf.add_image(os.path.join(compare_images_folder, img_file), caption=img_file.replace("_", " ").replace(".png", ""))

# Save PDF Report
report_path = os.path.join(comparison_folder, "comparison_report.pdf")
pdf.output(report_path)
print(f"Report saved to {report_path}")
