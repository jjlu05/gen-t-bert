import time
import glob
import pandas as pd
import math
import argparse
import table_integration, alite_fd_original
from tqdm import tqdm
import os
import json
import sys
sys.path.append('../discovery/')
from evaluatePaths import setTDR, bestMatchingTuples, instanceSimilarity
from calcDivergence import table_Dkl, getQueryConditionalVals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="tptr", choices=[
        'tptr', 'santos_large_tptr', 'tptr_groundtruth', 'tptr_small', 'tptr_large',
        't2d_gold', 'TUS_t2d_gold', 'wdc_t2d_gold'])
    parser.add_argument("--timeout", type=int, default=25263)
    parser.add_argument("--genT", type=int, default=1)
    parser.add_argument("--doPS", type=int, default=1)

    hp = parser.parse_args()
    benchmark = hp.benchmark

    base_dir = os.path.join("datasets", benchmark)
    query_path = os.path.join(base_dir, "queries")
    output_dir = "output_tables/%s/" % benchmark
    if hp.genT:
        output_dir = "genT_output_tables/%s/" % benchmark
    elif hp.doPS:
        output_dir = "output_tables_projSel/%s/" % benchmark

    os.makedirs(output_dir, exist_ok=True)
    print("========= Benchmark:", benchmark)

    if '_groundtruth' in benchmark:
        query_path = os.path.join("datasets", benchmark.split('_groundtruth')[0], "queries")

    datasets = glob.glob(os.path.join(query_path, '*.csv'))

    allTDR_recall, allTDR_prec, allInstanceSim, allDkl = {}, {}, {}, {}
    allRuntimes = {}
    numSources = 0
    saved_sources, timedOutSources = [], []
    avgSizeOutput, avgSizeRatio = [], []
    individual_prec_recall, runtimes = {}, {}

    originating_table_path = os.path.join("results_candidate_tables", benchmark, "originatingTables.json")
    allOriginatingTableDict = None
    if os.path.isfile(originating_table_path):
        with open(originating_table_path) as f:
            allOriginatingTableDict = json.load(f)
    elif hp.genT:
        print("Need to Generate Originating Tables before finishing Gen-T")

    algStartTime = time.time()
    print("Using Gen-T" if hp.genT else "Using ALITE Baseline")

    for indx in tqdm(range(len(datasets))):
        source_table = os.path.basename(datasets[indx])
        if allOriginatingTableDict and source_table not in allOriginatingTableDict:
            continue
        sourceDf = pd.read_csv(os.path.join(query_path, source_table))
        print(f"\t=========== {indx}) Source Table: {source_table} ===========")
        print(f"Source has {sourceDf.shape[1]} cols, {sourceDf.shape[0]} rows --> {sourceDf.shape[0]*sourceDf.shape[1]} total values")

        primaryKey = sourceDf.columns.tolist()[0]
        foreignKeys = [col for col in sourceDf.columns if 'key' in col and col != primaryKey]
        if 't2d_gold' in benchmark:
            if all(pd.isna(val) for val in sourceDf[primaryKey].values):
                for col in sourceDf.columns[1:]:
                    if any(not pd.isna(val) for val in sourceDf[col].values):
                        primaryKey = col
                        break
            foreignKeys = []

        queryValPairs = getQueryConditionalVals(sourceDf, primaryKey)
        startTime = time.time()

        if hp.genT:
            timed_out, noCandidates, numOutputVals = table_integration.main(
                benchmark, source_table, allOriginatingTableDict[source_table], hp.timeout)
        else:
            candidate_path = os.path.join("results_candidate_tables", benchmark, "candidateTables.json")
            with open(candidate_path) as f:
                allCandidateTableDict = json.load(f)
            timed_out, noCandidates, numOutputVals = alite_fd_original.main(
                benchmark, source_table, allCandidateTableDict[source_table], hp.doPS, hp.timeout)

        if timed_out:
            print(f"\t\t\tAlite Timed out for Source Table {source_table} after {time.time() - startTime:.3f} seconds")
            timedOutSources.append(source_table)
            continue
        if noCandidates:
            print(f"\t\t\tAlite Has No Candidates for Source Table {source_table} after {time.time() - startTime:.3f} seconds")
            continue

        print(f"\t\t\tAlite Finished for {source_table} in {time.time() - startTime:.3f} sec, Output size: {numOutputVals}")
        runtimes[source_table] = time.time() - startTime
        avgSizeOutput.append(numOutputVals)
        avgSizeRatio.append(numOutputVals / (sourceDf.shape[0] * sourceDf.shape[1]))

        fd_result = pd.read_csv(os.path.join(output_dir, source_table))
        TDR_recall, TDR_precision = setTDR(sourceDf, fd_result)
        bestMatchingDf = bestMatchingTuples(sourceDf, fd_result, primaryKey)
        if bestMatchingDf is None:
            continue

        instanceSim = instanceSimilarity(sourceDf, bestMatchingDf, primaryKey)
        bestMatchingDf = bestMatchingDf[sourceDf.columns]
        tableKL, _, _ = table_Dkl(sourceDf, bestMatchingDf, primaryKey, queryValPairs, math.inf, log=1)

        print(f"=== EVAL: Recall={TDR_recall:.3f}, Precision={TDR_precision:.3f}, Sim={instanceSim:.3f}, KL={tableKL:.3f}")

        try:
            f1_score = 2 * (TDR_precision * TDR_recall) / (TDR_precision + TDR_recall)
        except:
            f1_score = 0
        individual_prec_recall[source_table] = {'Precision': TDR_precision, 'Recall': TDR_recall, 'F1_Score': f1_score}

        metrics = [TDR_recall, TDR_precision, instanceSim, tableKL, time.time() - startTime]
        category = "all"
        if 'tptr' in benchmark:
            if 'psql_many' in source_table:
                category = 'manyJoins'
            elif 'psql_edge' in source_table:
                category = 'simple'
            else:
                category = 'oneJoin'

        for idx, m_dict in enumerate([allTDR_recall, allTDR_prec, allInstanceSim, allDkl, allRuntimes]):
            m_dict.setdefault(category, []).append(metrics[idx])
            m_dict.setdefault('all', []).append(metrics[idx])

        saved_sources.append(source_table)
        numSources += 1

    print("\t\t\t=================================")
    stats_dir = os.path.join("experiment_logs", benchmark)
    os.makedirs(stats_dir, exist_ok=True)

    summary = {
        'num_sources': [numSources, saved_sources],
        'timed_out_sources': [len(timedOutSources), timedOutSources]
    }

    metrics_map = {
        'TDR_Recall': allTDR_recall,
        'TDR_Precision': allTDR_prec,
        'Instance_Similarity': allInstanceSim,
        'Instance_Divergence': {k: [1 - v for v in vs] for k, vs in allInstanceSim.items()},
        'KL_Divergence': allDkl,
        'Runtimes': allRuntimes
    }

    for name, metric in metrics_map.items():
        summary[name] = {k: round(sum(v) / len(v), 3) for k, v in metric.items()}

    summary['ouptut_size'] = round(sum(avgSizeOutput) / len(avgSizeOutput), 3)
    summary['size_ratio'] = round(sum(avgSizeRatio) / len(avgSizeRatio), 3)

    if hp.genT:
        with open(os.path.join(stats_dir, "final_genT_results.json"), "w") as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(stats_dir, "each_source_result_genT.json"), "w") as f:
            json.dump(individual_prec_recall, f, indent=4)
    elif hp.doPS:
        with open(os.path.join(stats_dir, "final_alitePS_results.json"), "w") as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(stats_dir, "each_source_result_alitePS.json"), "w") as f:
            json.dump(individual_prec_recall, f, indent=4)
        with open(os.path.join(stats_dir, "runtimes_alitePS.json"), "w") as f:
            json.dump(runtimes, f, indent=4)
    else:
        with open(os.path.join(stats_dir, "final_alite_results.json"), "w") as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(stats_dir, "each_source_result_alite.json"), "w") as f:
            json.dump(individual_prec_recall, f, indent=4)
        with open(os.path.join(stats_dir, "runtimes_alite.json"), "w") as f:
            json.dump(runtimes, f, indent=4)

    print(f"FINISHED ALL {numSources} SOURCES IN {(time.time() - algStartTime):.3f} sec")
