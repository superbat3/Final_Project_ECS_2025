import pandas as pd
import re
from collections import Counter
import sacrebleu
from bert_score import score
import transformers
transformers.logging.set_verbosity_error()


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def f1_score(pred, gold):
    pred_tokens = normalize(pred)
    gold_tokens = normalize(gold)
    common = (Counter(pred_tokens) & Counter(gold_tokens))
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# =============== evaluate F1 score ============
df = pd.read_csv("pilot_results.csv")

df["f1"] = df.apply(lambda r: f1_score(r["final_answer"], r["gold"]), axis=1)

print("\nAverage F1:")
print(df.groupby("prompt_style")["f1"].mean())


# =============== evaluate bleu ===============
def bleu_score(pred, gold):
    return sacrebleu.sentence_bleu(pred, [gold]).score

df["bleu"] = df.apply(lambda r: bleu_score(r["final_answer"], r["gold"]), axis=1)

print("\nAverage BLEU:")
print(df.groupby("prompt_style")["bleu"].mean())


# =============== evaluate BERTScore ============
def clean(s):
    if not isinstance(s, str):
        return ""
    return s.replace("\n", " ").replace("**", "")

df["final_answer"] = df["final_answer"].apply(clean)
df["gold"] = df["gold"].apply(clean)

bert_f1 = []

for i, r in df.iterrows():
    try:
        _, _, F1 = score([r["final_answer"]], [r["gold"]], lang="en", verbose=False)
        bert_f1.append(F1.item())
    except Exception as e:
        print(f"Error on row {i}: {e}")
        bert_f1.append(None)

df["bert_f1"] = bert_f1
df["bert_f1"] = df["bert_f1"].fillna(0)

print(df.groupby("prompt_style")["bert_f1"].mean())


df.to_csv("pilot_results_with_scores.csv", index=False)
print("\nSaved to pilot_results_with_scores.csv")