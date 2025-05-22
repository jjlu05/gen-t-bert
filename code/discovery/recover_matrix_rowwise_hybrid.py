
import pandas as pd
import numpy as np
from bert_utils import bert_similarity

def get_row_signature(row):
    return " ".join(str(v) for v in row if pd.notnull(v))

def hybrid_score_table(source_df, candidate_df, row_threshold=0.85):
    source_rows = [row for _, row in source_df.iterrows()]
    candidate_rows = [row for _, row in candidate_df.iterrows()]
    matched = 0
    total_score = 0.0

    for src_row in source_rows:
        src_sig = get_row_signature(src_row)
        best_score = -1
        best_cand = None

        for cand_row in candidate_rows:
            cand_sig = get_row_signature(cand_row)
            sim = bert_similarity(src_sig, cand_sig)
            if sim > best_score:
                best_score = sim
                best_cand = cand_row

        if best_score >= row_threshold:
            matched += 1
            # Do cell-by-cell validation
            src_vals = [str(v).strip().lower() for v in src_row if pd.notnull(v)]
            cand_vals = [str(v).strip().lower() for v in best_cand if pd.notnull(v)]
            matched_cells = sum(1 for v in src_vals if v in cand_vals)
            total_score += matched_cells / max(len(src_vals), 1)

    return matched / len(source_rows), total_score / max(matched, 1) if matched > 0 else 0.0
