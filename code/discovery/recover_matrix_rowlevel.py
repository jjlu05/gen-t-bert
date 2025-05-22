import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bert_utils import bert_similarity

def get_row_signature(row):
    return " ".join(str(v) for v in row if pd.notnull(v))

def score_table_rowwise(source_df, candidate_df, threshold=0.85):
    source_sigs = [get_row_signature(row) for _, row in source_df.iterrows()]
    candidate_sigs = [get_row_signature(row) for _, row in candidate_df.iterrows()]

    matched = 0
    for src_sig in source_sigs:
        best_score = max([bert_similarity(src_sig, cand_sig) for cand_sig in candidate_sigs])
        if best_score >= threshold:
            matched += 1

    return matched / len(source_sigs) if source_sigs else 0.0
