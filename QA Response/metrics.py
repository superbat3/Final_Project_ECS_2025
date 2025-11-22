import pandas as pd
import json
import sacrebleu
from bert_score import score as bert_score
from typing import List
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def clean(text: str) -> str:
    text = re.sub(r'\*\*', '', text) 
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'["\'`\*:]', '', text) 
    text = ' '.join(text.split())
    
    return text

INPUT_FILE = "data/answer/4omini-direct.jsonl"
OUTPUT_FILE = "data/metrics/4omini-direct.csv"


# read records from JSONL
records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

pred_list = [clean(r["model_answer"]) for r in records]
gold_list = [clean(r["gold_answer"]) for r in records]


def compute_f1(pred, gold):
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# f1, bleu, bert calculation
f1_list = [compute_f1(p, g) for p, g in zip(pred_list, gold_list)]
bleu_list = [sacrebleu.sentence_bleu(p, [g]).score for p, g in zip(pred_list, gold_list)] 
P, R, F1_scores = bert_score(pred_list, gold_list, lang="en") 
bert_list = F1_scores.tolist()


# rouge calculation
rouge1_list = []
rougeL_list = []

# initialize ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

for pred, gold in zip(pred_list, gold_list):
    scores = rouge_scorer_obj.score(gold, pred)
    rouge1_list.append(scores['rouge1'].fmeasure)
    rougeL_list.append(scores['rougeL'].fmeasure)


# cosine similarity calculation
cosine_sim_list = []

# initialize Sentence-Transformer model for Cosine Similarity
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    print("Warning: Could not load SentenceTransformer model. Cosine Similarity will not be calculated.")
    sbert_model = None

if sbert_model is not None:
    # generate embeddings
    pred_embeddings = sbert_model.encode(pred_list, convert_to_tensor=False)
    gold_embeddings = sbert_model.encode(gold_list, convert_to_tensor=False)
    
    # calculate cosine similarity
    for pred_emb, gold_emb in zip(pred_embeddings, gold_embeddings):
        # reshape to 2D arrays
        sim = cosine_similarity(
            pred_emb.reshape(1, -1), 
            gold_emb.reshape(1, -1)
        )[0][0]
        cosine_sim_list.append(sim)
else:
    cosine_sim_list = [0.0] * len(pred_list) 



# add computed scores back to records
records_with_scores = []

for r, f1, bleu, bert, r1, rL, cos_sim in zip(
    records, f1_list, bleu_list, bert_list, rouge1_list, rougeL_list, cosine_sim_list
):
    r_copy = r.copy() 
    r_copy.update({
        "token_f1": f1,
        "sacrebleu": bleu,
        "bertscore_f1": bert,
        "rouge1_f1": r1,          
        "rougeL_f1": rL,         
        "cosine_similarity": cos_sim
    })
    records_with_scores.append(r_copy)

# save to CSV
df = pd.DataFrame(records_with_scores)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"{OUTPUT_FILE} has been saved.")