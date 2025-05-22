import os
import json

benchmark = "santos_benchmark"
query_dir = os.path.join("datasets", benchmark, "query")
lake_dir = os.path.join("datasets", benchmark, "datalake")

query_files = [f for f in os.listdir(query_dir) if f.endswith(".csv")]
lake_files = [f for f in os.listdir(lake_dir) if f.endswith(".csv")]

candidates = { query: lake_files for query in query_files }

output_path = f"results_candidate_tables/{benchmark}/candidateTables.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(candidates, f, indent=4)

print(f"✅ Wrote candidateTables.json for {len(query_files)} queries, each with {len(lake_files)} candidates.")
