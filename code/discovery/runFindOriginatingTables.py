"""
Iterate through (1) findCandidates and (2) discoveryGraph until we find a path with KL-divergence of 0
"""
import os
import sys
import time
import json
import glob
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "findCandidates")))
import set_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import recover_matrix_ternary

DATA_ROOT = os.path.join("datasets")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_bert", action="store_true", help="Use BERT for value similarity")

    parser.add_argument('--benchmark', type=str, required=True,
                    choices=['tptr', 'santos_large_tptr', 'tptr_groundtruth', 'tptr_small', 'tptr_large',
                             't2d_gold', 'TUS_t2d_gold', 'wdc_t2d_gold', 'tptr_small_bert', 'tptr_bert_hard'])

    parser.add_argument("--saveMT", type=int, default=1)
    hp = parser.parse_args()

    runStarmie = (hp.benchmark == 'santos_large_tptr')

    query_dir = os.path.join(DATA_ROOT, hp.benchmark, "queries")
    datasets = glob.glob(os.path.join(query_dir, "*.csv"))

    candidate_table_path = os.path.join("results_candidate_tables", hp.benchmark, "candidateTables.json")
    with open(candidate_table_path) as json_file:
        allCandidateTableDict = json.load(json_file)

    avgRuntimes = {k: [] for k in ['matrix_initialization', 'matrix_traversal', 'all']}
    eachRuntimes = {}
    numSources = 0
    algStartTime = time.time()
    correct_count = 0
    total_count = 0

    allOriginsForSources = {}
    for indx in tqdm(range(len(datasets))):
        source_table = os.path.basename(datasets[indx])
        print(f"\n=========== {indx}) Source Table: {source_table} ===========")
        currStartTime = time.time()

        originating_tables, ternMatrixRuntimes = recover_matrix_ternary.main(hp.benchmark, source_table, use_bert=hp.use_bert)
        # Load ground truth mapping
        with open(os.path.join("results_candidate_tables", hp.benchmark, "originatingTables.json")) as f:
            ground_truth = json.load(f)

        # Get predicted table (top match from Gen-T's traversal)
        predicted_tables = originating_tables if originating_tables else []
        predicted_top = predicted_tables[0] if predicted_tables else None

        # Get correct table
        true_table = list(ground_truth[source_table].keys())[0]

        # Compare and print result
        if predicted_top == true_table:
            print(f"CORRECT: {source_table} → {predicted_top}")
        else:
            print(f"INCORRECT: {source_table} → {predicted_top}, expected {true_table}")
        total_count += 1
        if predicted_top == true_table:
            correct_count += 1

        if originating_tables:
            allOriginsForSources[source_table] = {
                t: allCandidateTableDict[source_table][t]
                for t in originating_tables
            }

        if not ternMatrixRuntimes:
            print("There were no candidates found in TERNARY MATRIX TRAVERSAL")
            continue

        avgRuntimes['matrix_initialization'] += ternMatrixRuntimes['matrix_initialization']
        avgRuntimes['matrix_traversal'] += ternMatrixRuntimes['matrix_traversal']
        avgRuntimes['all'].append(time.time() - currStartTime)

        eachRuntimes[source_table] = {
            'matrix_initialization': ternMatrixRuntimes['matrix_initialization'],
            'matrix_traversal': ternMatrixRuntimes['matrix_traversal'],
            'all': time.time() - currStartTime
        }

        numSources += 1

    if hp.saveMT:
        origin_table_path = os.path.join("results_candidate_tables", hp.benchmark, "originatingTables.json")
        try:
            os.remove(origin_table_path)
        except FileNotFoundError:
            pass
        with open(origin_table_path, "w") as f:
            json.dump(allOriginsForSources, f, indent=4)

    print("\n============================================")

    log_dir = os.path.join("experiment_logs", hp.benchmark)
    os.makedirs(log_dir, exist_ok=True)

    sourceStats = {'num_sources': numSources}
    for k, v_list in avgRuntimes.items():
        sourceStats[f'avg_{k}'] = sum(v_list) / len(v_list) if v_list else 0.0
    sourceStats.update(eachRuntimes)

    with open(os.path.join(log_dir, "runtimes_genT.json"), "w") as f:
        json.dump(sourceStats, f, indent=4)
    print(f"\n✅ FINAL TOP-1 ACCURACY: {correct_count}/{total_count} = {correct_count / total_count:.2%}")

    print("FINISHED ALL %d SOURCES IN %.3f seconds (%.2f min, %.2f hrs)" % (
        numSources,
        time.time() - algStartTime,
        (time.time() - algStartTime) / 60,
        (time.time() - algStartTime) / 3600
    ))
