"""
Iterate through (1) findCandidates and (2) discoveryGraph until we find a path with KL-divergence of 0
"""
import time
import os
import glob
import pandas as pd
import sys
from tqdm import tqdm

# Import internal modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "findCandidates")))
import set_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import recover_matrix_ternary

# ✅ Data folder root
DATA_ROOT = os.path.join("datasets")

def get_lake(benchmark):
    '''
    Get data lake tables and convert values to string format.
    '''
    datalake_path = os.path.join(DATA_ROOT, benchmark, "datalake")
    totalLakeTables = glob.glob(os.path.join(datalake_path, "*.csv"))

    rawLakeDfs = {}
    allLakeTableCols = {}

    for filepath in totalLakeTables:
        table = os.path.basename(filepath)
        df = pd.read_csv(filepath, lineterminator="\n")
        rawLakeDfs[table] = df

        for col in df.columns:
            if table not in allLakeTableCols:
                allLakeTableCols[table] = {}
            allLakeTableCols[table][col] = [str(val).strip() for val in df[col] if not pd.isna(val)]

    return rawLakeDfs, allLakeTableCols

if __name__ == '__main__':
    # Set benchmark here
    benchmark = 'tptr'  # change to 'tptr_small' if you're using the mock dataset
    runStarmie = 1 if benchmark == 'santos_large_tptr' else 0
    saveMT = 1

    # ✅ Queries folder
    query_path = os.path.join(DATA_ROOT, benchmark, "queries")
    datasets = glob.glob(os.path.join(query_path, "*.csv"))

    allTernTDR_recall = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allTernTDR_prec = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allTernInstanceSim = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allTernDkl = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}
    allTernRuntimes = {k: [] for k in ['matrix_initialization', 'matrix_traversal', 'all']}
    ternRuntimes = {k: [] for k in ['simple', 'oneJoin', 'manyJoins', 'all']}

    numSources = 0
    algStartTime = time.time()

    print("\n=========== GETTING DATA LAKE TABLES ===========")
    lakeDfs, allLakeTableCols = get_lake(benchmark)
    print(f"{len(lakeDfs)} lakeDfs")

    for indx in tqdm(range(len(datasets))):
        source_table = os.path.basename(datasets[indx])
        threshold = 0.2
        print(f"\n=========== {indx}) Source Table: {source_table} ===========")

        # ---- Set Similarity Phase ----
        print("=========== BEGIN SET SIMILARITY ===========")
        setSimTime = time.time()
        noCandidates = set_similarity.main(benchmark, source_table, threshold, lakeDfs, allLakeTableCols, includeStarmie=runStarmie)
        if noCandidates:
            print("No candidates found (no candidate for key)")
            continue
        print(f"=========== END SET SIMILARITY FOR {source_table} IN {time.time() - setSimTime:.3f} sec ===========\n")

        # ---- Ternary Matrix Phase ----
        print("=========== BEGIN TERNARY MATRIX TRAVERSAL ===========")
        currStartTime = time.time()
        ternMatrixTDR_recall, ternMatrixTDR_prec, ternInstanceSim, ternTableDkl, numOutputVals, ternMatrixRuntimes = recover_matrix_ternary.main(
            benchmark, source_table, saveTraversedTables=saveMT)

        if not ternMatrixRuntimes:
            print("No candidates found in TERNARY MATRIX TRAVERSAL")
            continue

        allTernRuntimes['matrix_initialization'] += ternMatrixRuntimes['matrix_initialization']
        allTernRuntimes['matrix_traversal'] += ternMatrixRuntimes['matrix_traversal']
        allTernRuntimes['all'].append(time.time() - currStartTime)

        print(f"=========== END TERNARY MATRIX TRAVERSAL in {(time.time() - currStartTime):.3f} seconds ===========")
        numSources += 1

    print("\n=================================")
    print(f"FINISHED ALL {numSources} SOURCES IN {time.time() - algStartTime:.3f} seconds "
          f"({(time.time() - algStartTime)/60:.3f} minutes, {(time.time() - algStartTime)/3600:.3f} hrs)")

    if numSources > 0:
        print(f"Average Total Runtime: {sum(allTernRuntimes['all'])/numSources:.3f} sec "
              f"({(sum(allTernRuntimes['all'])/numSources)/60:.3f} min)")
        print(f"Average matrix_initialization: {sum(allTernRuntimes['matrix_initialization'])/numSources:.3f} sec")
        print(f"Average matrix_traversal: {sum(allTernRuntimes['matrix_traversal'])/numSources:.3f} sec")
