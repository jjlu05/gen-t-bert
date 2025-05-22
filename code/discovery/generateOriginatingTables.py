# GenerateOriginatingTables.py

import pandas as pd
import json
import os

# Set your benchmark name here
benchmark = "santos_benchmark"

# Path to the CSV ground truth file
csv_path = f"datasets/{benchmark}/santos_small_benchmark_groundtruth.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Convert to the Gen-T JSON structure
originating_tables = {}
for _, row in df.iterrows():
    query = row["query_table"]
    lake = row["data_lake_table"]
    if query not in originating_tables:
        originating_tables[query] = {}
    originating_tables[query][lake] = {}

# Write to JSON file
output_path = f"results_candidate_tables/{benchmark}/originatingTables.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(originating_tables, f, indent=4)

print(f"✅ Rebuilt originatingTables.json with {len(originating_tables)} queries")
