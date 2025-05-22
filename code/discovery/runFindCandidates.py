import time
import os
import glob
import pandas as pd
import argparse
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "findCandidates")))
import set_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import recover_matrix_ternary

DATA_ROOT = os.path.join("datasets")

def get_lake(benchmark):
    '''
    Load all data lake tables and their stringified column values
    '''
    datalake_path = os.path.join(DATA_ROOT, benchmark, "datalake")
    lakeFiles = glob.glob(os.path.join(datalake_path, "*.csv"))

    rawLakeDfs = {}
    allLakeTableCols = {}
    for filepath in lakeFiles:
        table = os.path.basename(filepath)
        df = pd.read_csv(filepath, lineterminator="\n")
        rawLakeDfs[table] = df
        allLakeTableCols[table] = {
            col: [str(val).strip() for val in df[col] if not pd.isna(val)]
            for col in df.columns
        }
    return rawLakeDfs, allLakeTableCols

def get_starmie_candidates(benchmark):
    '''
    Load Starmie-discovered candidate tables (if applicable)
    '''
    starmie_path = os.path.join("..", "..", "Starmie_candidate_results", benchmark, "starmie_candidates.json")
    with open(starmie_path) as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="tptr_small")
    parser.add_argument("--threshold", type=float, default=0.2)
    hp = parser.parse_args()

    runStarmie = (hp.benchmark == 'santos_large_tptr')

    queries_path = os.path.join(DATA_ROOT, hp.benchmark, "queries")
    datasets = glob.glob(os.path.join(queries_path, "*.csv"))

    print("\n=========== GETTING DATA LAKE TABLES ===========")
    lakeDfs, allLakeTableCols = get_lake(hp.benchmark)
    print(f"{len(lakeDfs)} lakeDfs loaded")

    if runStarmie:
        starmie_candidates = get_starmie_candidates(hp.benchmark)

    output_dir = os.path.join("results_candidate_tables", hp.benchmark)
    os.makedirs(output_dir, exist_ok=True)

    allCandidatesForSources = {}

    for i, filepath in enumerate(tqdm(datasets)):
        source_table = os.path.basename(filepath)
        print(f"\n=========== {i}) Source Table: {source_table} ===========")
        source_candidates = starmie_candidates.get(source_table.replace(".csv", ""), []) if runStarmie else list(lakeDfs.keys())

        candidateTablesFound, noCandidates = set_similarity.main(
            hp.benchmark,
            source_table,
            hp.threshold,
            lakeDfs,
            allLakeTableCols,
            source_candidates
        )

        if noCandidates:
            print("No candidates found (no primary key overlap)")
        allCandidatesForSources[source_table] = candidateTablesFound

    output_path = os.path.join(output_dir, "candidateTables.json")
    with open(output_path, "w") as f:
        json.dump(allCandidatesForSources, f, indent=4)
