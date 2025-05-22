import pandas as pd
import numpy as np
from bert_utils import bert_similarity

def get_row_signature(row):
    """Generate a string signature from a row's non-null values"""
def get_row_signature(row, ignore_keys=["company_id"]):
    return " ".join(str(v) for k, v in row.items() if pd.notnull(v) and k not in ignore_keys)

def enhanced_score_table(source_df, candidate_df, row_threshold=0.85, strict_validation=False):
    """
    Score table matches with both row-level and cell-level validation
    
    Args:
        source_df: Source dataframe
        candidate_df: Candidate dataframe
        row_threshold: Threshold for considering a row match (using BERT)
        strict_validation: If True, requires exact cell matches; if False, uses BERT for cell matching too
        
    Returns:
        tuple: (row_match_score, cell_match_score, combined_score)
    """
    # Step 1: Row-level matching (similar to current implementation)
    source_rows = [row for _, row in source_df.iterrows()]
    candidate_rows = [row for _, row in candidate_df.iterrows()]
    
    row_matches = []  # Will contain (source_idx, candidate_idx, score) tuples
    
    # Find best candidate row for each source row
    for src_idx, src_row in enumerate(source_rows):
        src_sig = get_row_signature(src_row)
        best_score = -1
        best_cand_idx = None
        
        for cand_idx, cand_row in enumerate(candidate_rows):
            cand_sig = get_row_signature(cand_row)
            sim = bert_similarity(src_sig, cand_sig)
            if sim > best_score:
                best_score = sim
                best_cand_idx = cand_idx
        
        if best_score >= row_threshold:
            row_matches.append((src_idx, best_cand_idx, best_score))
    
    # Calculate row match score
    row_match_score = len(row_matches) / len(source_rows) if source_rows else 0.0
    
    # Step 2: Cell-level validation for matched rows
    total_cells = 0
    matched_cells = 0
    
    matched_src_cols = set(source_df.columns).intersection(set(candidate_df.columns))
    
    for src_idx, cand_idx, _ in row_matches:
        src_row = source_rows[src_idx]
        cand_row = candidate_rows[cand_idx]
        
        for col in matched_src_cols:
            src_val = source_df.iloc[src_idx][col]
            cand_val = candidate_df.iloc[cand_idx][col]
            
            # Skip null values
            if pd.isna(src_val) or pd.isna(cand_val):
                continue
                
            total_cells += 1
            
            # Either exact match or BERT similarity based on strict_validation flag
            if strict_validation:
                if str(src_val).strip().lower() == str(cand_val).strip().lower():
                    matched_cells += 1
            else:
                cell_sim = bert_similarity(str(src_val), str(cand_val))
                if cell_sim >= 0.85:  # Using same threshold for cell matching
                    matched_cells += 1
    
    # Calculate cell match score
    cell_match_score = matched_cells / total_cells if total_cells > 0 else 0.0
    
    # Combined score (weighted average - can be adjusted)
    combined_score = 0.6 * row_match_score + 0.4 * cell_match_score
    
    return row_match_score, cell_match_score, combined_score


def generate_alignment_report(source_df, candidate_df, row_threshold=0.85):
    """
    Generate a detailed alignment report between source and candidate tables
    
    Returns:
        dict: Detailed information about the match
    """
    source_rows = [row for _, row in source_df.iterrows()]
    candidate_rows = [row for _, row in candidate_df.iterrows()]
    row_matches = []
    
    # Find matches
    for src_idx, src_row in enumerate(source_rows):
        src_sig = get_row_signature(src_row)
        best_score = -1
        best_cand_idx = None
        
        for cand_idx, cand_row in enumerate(candidate_rows):
            cand_sig = get_row_signature(cand_row)
            sim = bert_similarity(src_sig, cand_sig)
            if sim > best_score:
                best_score = sim
                best_cand_idx = cand_idx
        
        if best_score >= row_threshold:
            row_matches.append((src_idx, best_cand_idx, best_score))
    
    # Generate column alignment scores
    common_cols = set(source_df.columns).intersection(set(candidate_df.columns))
    col_scores = {}
    
    for col in common_cols:
        matched_values = 0
        total_values = 0
        
        for src_idx, cand_idx, _ in row_matches:
            src_val = source_df.iloc[src_idx][col]
            cand_val = candidate_df.iloc[cand_idx][col]
            
            if pd.isna(src_val) or pd.isna(cand_val):
                continue
                
            total_values += 1
            sim = bert_similarity(str(src_val), str(cand_val))
            if sim >= 0.85:
                matched_values += 1
        
        col_scores[col] = matched_values / total_values if total_values > 0 else 0.0
    
    # Calculate overall scores
    row_score = len(row_matches) / len(source_rows) if source_rows else 0.0
    cell_score = sum(col_scores.values()) / len(col_scores) if col_scores else 0.0
    combined_score = 0.6 * row_score + 0.4 * cell_score
    
    return {
        "matched_rows": len(row_matches),
        "total_source_rows": len(source_rows),
        "row_score": row_score,
        "column_scores": col_scores,
        "cell_score": cell_score,
        "combined_score": combined_score,
        "row_matches": row_matches  # (src_idx, cand_idx, score) tuples
    }