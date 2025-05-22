"""
Iterate through (1) findCandidates and (2) discoveryGraph until we find a path with KL-divergence of 0
Enhanced with cell-level validation after row matching
Debug version with extensive logging
"""
import os
import sys
import time
import json
import glob
import argparse
import pandas as pd
from tqdm import tqdm
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"debug_santos_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)  # Output to console as well
    ]
)
logger = logging.getLogger(__name__)

# Import your modules with error handling
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "findCandidates")))
    import set_similarity
    
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import recover_matrix_ternary
    
    import recover_matrix_rowlevel
    from enhanced_matrix_rowlevel import enhanced_score_table, generate_alignment_report
    
    logger.info("Successfully imported all required modules")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

DATA_ROOT = os.path.join("datasets")

def safe_division(numerator, denominator):
    """Safe division function to avoid division by zero errors"""
    if denominator == 0:
        logger.warning(f"Division by zero prevented: {numerator}/{denominator}")
        return 0
    return numerator / denominator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_bert", action="store_true", help="Use BERT for value similarity")
    parser.add_argument("--validation_mode", type=str, default="hybrid", 
                        choices=["row_only", "hybrid", "strict"],
                        help="Validation mode: row_only (current), hybrid (BERT for cells), strict (exact cell matches)")
    parser.add_argument("--cell_weight", type=float, default=0.4,
                        help="Weight for cell-level validation (0.0-1.0)")
    
    parser.add_argument('--benchmark', type=str, required=True,
                    choices=['tptr', 'santos_benchmark', 'santos_large_tptr', 'tptr_groundtruth', 'tptr_small', 'tptr_large',
                             't2d_gold', 'TUS_t2d_gold', 'wdc_t2d_gold', 'tptr_small_bert', 'tptr_bert_hard'])

    parser.add_argument("--saveMT", type=int, default=1)
    parser.add_argument("--save_report", action="store_true", help="Save detailed matching report")
    hp = parser.parse_args()
    
    # Log command line arguments
    logger.info(f"Running with arguments: {vars(hp)}")

    runStarmie = (hp.benchmark == 'santos_large_tptr')

    # Check existence of directory and files
    query_dir = os.path.join(DATA_ROOT, hp.benchmark, "query")
    if not os.path.exists(query_dir):
        logger.error(f"Query directory does not exist: {query_dir}")
        sys.exit(1)
    
    logger.info(f"Looking for datasets in: {query_dir}")
     # Check candidate tables
    candidate_table_path = os.path.join("results_candidate_tables", hp.benchmark, "candidateTables.json")
    logger.info(f"Loading candidate tables from: {candidate_table_path}")
    
    if not os.path.exists(candidate_table_path):
        logger.error(f"Candidate table file does not exist: {candidate_table_path}")
        sys.exit(1)
        
    try:
        with open(candidate_table_path) as json_file:
            allCandidateTableDict = json.load(json_file)
        logger.info(f"Loaded {len(allCandidateTableDict)} entries from candidate tables file")
        
        # Log example entries from candidate tables
        if allCandidateTableDict:
            sample_key = next(iter(allCandidateTableDict))
            logger.info(f"Sample candidate key: {sample_key}")
            logger.info(f"Sample entry has {len(allCandidateTableDict[sample_key])} candidate tables")
    except Exception as e:
        logger.error(f"Error loading candidate tables: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    datasets = [os.path.join(query_dir, fname) for fname in allCandidateTableDict]
    logger.info(f"Found {len(datasets)} source tables")
    
    if len(datasets) == 0:
        logger.error(f"No CSV files found in {query_dir}. Please check your benchmark path.")
        sys.exit(1)
        
    # Log first few found datasets
    for i, ds in enumerate(datasets[:5]):
        logger.info(f"Dataset {i}: {os.path.basename(ds)}")
    
   
    

    # Check originating tables
    originating_table_path = os.path.join("results_candidate_tables", hp.benchmark, "originatingTables.json")
    logger.info(f"Checking originating tables at: {originating_table_path}")
    
    try:
        if os.path.exists(originating_table_path):
            with open(originating_table_path) as f:
                ground_truth = json.load(f)
            logger.info(f"Loaded {len(ground_truth)} entries from originating tables file")
            
            # Log example entries from ground truth
            if ground_truth:
                sample_key = next(iter(ground_truth))
                logger.info(f"Sample ground truth key: {sample_key}")
                logger.info(f"Sample entry: {ground_truth[sample_key]}")
        else:
            logger.warning(f"Originating tables file doesn't exist yet: {originating_table_path}")
    except Exception as e:
        logger.error(f"Error loading originating tables: {str(e)}")
        logger.error(traceback.format_exc())
        # Continue without ground truth if needed

    avgRuntimes = {k: [] for k in ['matrix_initialization', 'matrix_traversal', 'row_validation', 'cell_validation', 'all']}
    eachRuntimes = {}
    numSources = 0
    algStartTime = time.time()
    correct_count = 0
    total_count = 0

    allOriginsForSources = {}
    detailed_reports = {}
    
    try:
        for indx in tqdm(range(len(datasets))):
            try:
                source_table = os.path.basename(datasets[indx]).strip()
                
                logger.info(f"\n=========== {indx}) Source Table: {source_table} ===========")
                print(f"\n=========== {indx}) Source Table: {source_table} ===========")
                currStartTime = time.time()

                # Check if source table exists in candidate table dict
                if source_table not in allCandidateTableDict:
                    logger.warning(f"Source table {source_table} not found in candidate tables")
                    continue

                # Load source dataframe
                source_path = os.path.join("datasets", hp.benchmark, "query", source_table)
                logger.info(f"Loading source table from: {source_path}")
                
                try:
                    source_df = pd.read_csv(source_path)
                    logger.info(f"Source table loaded: {source_df.shape[0]} rows, {source_df.shape[1]} columns")
                    logger.debug(f"Source table columns: {source_df.columns.tolist()}")
                    logger.debug(f"Source table first row: {source_df.iloc[0].tolist() if len(source_df) > 0 else 'empty'}")
                except Exception as e:
                    logger.error(f"Error loading source table {source_table}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                if hp.validation_mode == "row_only" and hp.use_bert:
                    # Original row-wise BERT scoring only
                    logger.info("Using row_only validation mode with BERT")
                    row_validation_start = time.time()
                    table_scores = {}
                    
                    for cand_table, mapping in allCandidateTableDict[source_table].items():
                        path = os.path.join("datasets", hp.benchmark, "datalake", cand_table)
                        logger.debug(f"Checking candidate table: {cand_table}")
                        
                        if not os.path.exists(path):
                            logger.warning(f"Candidate table file not found: {path}")
                            continue
                        
                        try:
                            candidate_df = pd.read_csv(path)
                            logger.debug(f"Loaded candidate table: {candidate_df.shape[0]} rows, {candidate_df.shape[1]} columns")
                            score = recover_matrix_rowlevel.score_table_rowwise(source_df, candidate_df)
                            table_scores[cand_table] = score
                            logger.debug(f"Row-level score for {cand_table}: {score}")
                        except Exception as e:
                            logger.error(f"Error processing candidate {cand_table}: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue
                    
                    avgRuntimes['row_validation'].append(time.time() - row_validation_start)
                    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                    logger.info(f"Sorted candidates by row score: {sorted_tables[:3]}")

                    originating_tables = []

                    if not sorted_tables:
                        logger.warning(f"No valid candidate scores for {source_table}")
                    else:
                        # Pick top 3 candidates regardless of score
                        top_candidates = sorted_tables[:3]
                        validated_scores = {}

                        for tname, _ in top_candidates:
                            try:
                                candidate_df = pd.read_csv(os.path.join("datasets", hp.benchmark, "datalake", tname))
                                score = recover_matrix_ternary.evaluate_ternary_support(source_df, candidate_df, use_bert=True)
                                validated_scores[tname] = score
                                logger.debug(f"Ternary validation score for {tname}: {score}")
                            except Exception as e:
                                logger.error(f"Error in ternary validation for {tname}: {str(e)}")
                                logger.error(traceback.format_exc())
                                continue

                        # Sort by ternary matrix score
                        validated_sorted = sorted(validated_scores.items(), key=lambda x: x[1], reverse=True)
                        originating_tables = [t[0] for t in validated_sorted if validated_scores[t[0]] >= 0.6]
                        logger.info(f"Tables passing ternary validation: {originating_tables}")

                        if not originating_tables:
                            logger.warning(f"No tables passed ternary validation for {source_table}")

                elif hp.validation_mode in ["hybrid", "strict"]:
                    # Enhanced validation using both row and cell level comparison
                    logger.info(f"Using {hp.validation_mode} validation mode with cell_weight={hp.cell_weight}")
                    row_validation_start = time.time()
                    
                    table_scores = {}
                    table_detailed_scores = {}
                    
                    for cand_table, mapping in allCandidateTableDict[source_table].items():
                        path = os.path.join("datasets", hp.benchmark, "datalake", cand_table)
                        logger.debug(f"Checking candidate table: {cand_table}")
                        
                        if not os.path.exists(path):
                            logger.warning(f"Candidate table file not found: {path}")
                            continue
                        
                        try:
                            candidate_df = pd.read_csv(path)
                            logger.debug(f"Loaded candidate table: {candidate_df.shape[0]} rows, {candidate_df.shape[1]} columns")
                            
                            # First pass with row-level validation
                            row_score = recover_matrix_rowlevel.score_table_rowwise(source_df, candidate_df)
                            logger.debug(f"Row-level score for {cand_table}: {row_score}")
                            
                            # Skip cell validation if row score is too low
                            if row_score < 0.5:  # Can adjust this threshold
                                table_scores[cand_table] = row_score
                                continue
                                
                            # Cell validation start time
                            cell_validation_start = time.time()
                            
                            # Apply enhanced scoring with cell-level validation
                            strict_validation = (hp.validation_mode == "strict")
                            row_score, cell_score, combined_score = enhanced_score_table(
                                source_df, 
                                candidate_df,
                                strict_validation=strict_validation
                            )
                            
                            logger.debug(f"Cell-level validation for {cand_table}: row={row_score}, cell={cell_score}, combined={combined_score}")
                            
                            # Custom weighting based on parameter
                            combined_score = (1 - hp.cell_weight) * row_score + hp.cell_weight * cell_score
                            
                            table_scores[cand_table] = combined_score
                            
                            # Generate detailed report if requested
                            if hp.save_report:
                                table_detailed_scores[cand_table] = generate_alignment_report(source_df, candidate_df)
                            
                            avgRuntimes['cell_validation'].append(time.time() - cell_validation_start)
                        except Exception as e:
                            logger.error(f"Error processing candidate {cand_table}: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue
                    
                    avgRuntimes['row_validation'].append(time.time() - row_validation_start)
                    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                    logger.info(f"Sorted candidates: {sorted_tables[:3]}")
                    
                    # Tables with combined score above threshold
                    threshold = 0.85  # Can adjust this threshold too
                    logger.info(f"Final Scores for {source_table} (using threshold={threshold}, cell_weight={hp.cell_weight}):")
                    for name, score in sorted_tables[:5]:  # Display top 5
                        logger.info(f"  {name}: {score:.4f}")

                    originating_tables = [t[0] for t in sorted_tables if t[1] >= threshold]
                    logger.info(f"Selected originating tables: {originating_tables}")
                    ternMatrixRuntimes = None
                    
                    # Save detailed reports
                    if hp.save_report:
                        detailed_reports[source_table] = table_detailed_scores
                        
                else:
                    # Original matrix-based traversal approach
                    logger.info("Using original matrix-based traversal approach")
                    try:
                        originating_tables, ternMatrixRuntimes = recover_matrix_ternary.main(hp.benchmark, source_table, use_bert=hp.use_bert)
                        logger.info(f"Matrix traversal found tables: {originating_tables}")
                    except Exception as e:
                        logger.error(f"Error in matrix traversal for {source_table}: {str(e)}")
                        logger.error(traceback.format_exc())
                        originating_tables = []
                        ternMatrixRuntimes = None
                
                # --- Ground truth evaluation (shared) ---
                try:
                    with open(os.path.join("results_candidate_tables", hp.benchmark, "originatingTables.json")) as f:
                        ground_truth = json.load(f)
                    
                    predicted_tables = originating_tables if originating_tables else []
                    predicted_top = predicted_tables[0] if predicted_tables else None
                    
                    if source_table not in ground_truth:
                        logger.warning(f"Skipping {source_table} — not found in originatingTables.json")
                        continue
                        
                    logger.debug(f"Ground truth for {source_table}: {ground_truth[source_table]}")
                    true_table = list(ground_truth[source_table].keys())[0]

                    if predicted_top == true_table:
                        logger.info(f"CORRECT: {source_table} → {predicted_top}")
                    else:
                        logger.warning(f"INCORRECT: {source_table} → {predicted_top}, expected {true_table}")
                    
                    total_count += 1
                    if predicted_top == true_table:
                        correct_count += 1
                except Exception as e:
                    logger.error(f"Error in ground truth evaluation for {source_table}: {str(e)}")
                    logger.error(traceback.format_exc())

                if originating_tables:
                    allOriginsForSources[source_table] = {
                        t: allCandidateTableDict[source_table][t]
                        for t in originating_tables
                    }
                    
                # Track runtime for this source
                runtime = time.time() - currStartTime
                avgRuntimes['all'].append(runtime)
                
                if ternMatrixRuntimes:
                    avgRuntimes['matrix_initialization'] += ternMatrixRuntimes['matrix_initialization']
                    avgRuntimes['matrix_traversal'] += ternMatrixRuntimes['matrix_traversal']

                    eachRuntimes[source_table] = {
                        'matrix_initialization': ternMatrixRuntimes['matrix_initialization'],
                        'matrix_traversal': ternMatrixRuntimes['matrix_traversal'],
                        'all': runtime
                    }
                else:
                    eachRuntimes[source_table] = {
                        'row_validation': avgRuntimes['row_validation'][-1] if 'row_validation' in avgRuntimes and avgRuntimes['row_validation'] else 0,
                        'cell_validation': avgRuntimes['cell_validation'][-1] if 'cell_validation' in avgRuntimes and avgRuntimes['cell_validation'] else 0,
                        'all': runtime
                    }

                numSources += 1
            except Exception as e:
                logger.error(f"Unhandled exception processing table {indx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Save originating tables
        if hp.saveMT and allOriginsForSources:
            origin_table_path = os.path.join("results_candidate_tables", hp.benchmark, "originatingTables.json")
            try:
                os.remove(origin_table_path)
            except FileNotFoundError:
                pass
            
            with open(origin_table_path, "w") as f:
                json.dump(allOriginsForSources, f, indent=4)
                logger.info(f"Saved originating tables to {origin_table_path}")
                
        # Save detailed matching reports if requested
        if hp.save_report and detailed_reports:
            report_path = os.path.join("results_candidate_tables", hp.benchmark, "detailed_matching_reports.json")
            with open(report_path, "w") as f:
                json.dump(detailed_reports, f, indent=4)
                logger.info(f"Saved detailed reports to {report_path}")

        logger.info("\n============================================")
        print("\n============================================")

        log_dir = os.path.join("experiment_logs", hp.benchmark)
        os.makedirs(log_dir, exist_ok=True)

        sourceStats = {'num_sources': numSources}
        for k, v_list in avgRuntimes.items():
            if v_list:  # Only calculate average if there are values
                sourceStats[f'avg_{k}'] = sum(v_list) / len(v_list)
        sourceStats.update(eachRuntimes)

        # Save runtime statistics
        with open(os.path.join(log_dir, f"runtimes_genT_{hp.validation_mode}.json"), "w") as f:
            json.dump(sourceStats, f, indent=4)
            logger.info(f"Saved runtime statistics to {os.path.join(log_dir, f'runtimes_genT_{hp.validation_mode}.json')}")
            
        # Calculate and display accuracy with safe division
        accuracy = safe_division(correct_count, total_count)
        accuracy_str = f"{accuracy:.2%}" if total_count > 0 else "N/A (no tables processed)"
        
        logger.info(f"\n FINAL TOP-1 ACCURACY: {correct_count}/{total_count} = {accuracy_str}")
        print(f"\n FINAL TOP-1 ACCURACY: {correct_count}/{total_count} = {accuracy_str}")

        total_runtime = time.time() - algStartTime
        logger.info("FINISHED ALL %d SOURCES IN %.3f seconds (%.2f min, %.2f hrs)" % (
            numSources,
            total_runtime,
            total_runtime / 60,
            total_runtime / 3600
        ))
        print("FINISHED ALL %d SOURCES IN %.3f seconds (%.2f min, %.2f hrs)" % (
            numSources,
            total_runtime,
            total_runtime / 60,
            total_runtime / 3600
        ))
        
        # Print validation mode summary
        logger.info(f"\nValidation Mode: {hp.validation_mode}")
        print(f"\nValidation Mode: {hp.validation_mode}")
        if hp.validation_mode in ["hybrid", "strict"]:
            logger.info(f"Cell Weight: {hp.cell_weight:.2f}")
            print(f"Cell Weight: {hp.cell_weight:.2f}")
    
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Critical error: {str(e)}. See log file for details.")