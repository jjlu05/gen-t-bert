from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Optional: cache for performance
bert_cache = {}

def get_bert_embedding(text):
    text = str(text).lower().strip()
    if text in bert_cache:
        return bert_cache[text]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    bert_cache[text] = emb
    return emb
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

from knowledge_graph import get_related_entities

def bert_similarity(text1, text2):
    '''Enhanced BERT similarity using KG-aware expansion'''
    with torch.no_grad():
        tokens1 = tokenizer(text1, return_tensors='pt', truncation=True, padding=True)
        tokens2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=True)

        output1 = model(**tokens1).last_hidden_state[:, 0, :]
        output2 = model(**tokens2).last_hidden_state[:, 0, :]

        sim = torch.nn.functional.cosine_similarity(output1, output2).item()

    # Boost score if KG says they're semantically related
    text1_related = get_related_entities(text1)
    text2_related = get_related_entities(text2)

    if text2.lower() in text1_related or text1.lower() in text2_related:
        sim = min(1.0, sim + 0.2)  # small boost

    return sim
