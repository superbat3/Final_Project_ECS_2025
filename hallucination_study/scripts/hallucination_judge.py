import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Load API key from environment variable (.env)
load_dotenv() 

client = OpenAI()


def judge_hallucination(question, gold, answer):
    prompt = f"""
You are a factual accuracy evaluator for historical QA.

You will be given:
- a question,
- a gold (ground-truth) answer,
- a model-generated answer.

Your task:
Evaluate whether the model-generated answer contains any *hallucinated factual claims* not supported by the gold answer or known historical facts.

Scoring guidelines:
0   = No hallucination. All statements are supported by the gold answer or historically accurate.
0.5 = Minor hallucination. The answer is mostly correct but includes uncertain or unverified details not present in the gold answer.
1   = Major hallucination. The answer contains clear factual errors, wrong events, wrong people/time, or fabricated claims.

Only output the number 0, 0.5, or 1.
Do NOT explain your reasoning.

Question: {question}
Gold answer: {gold}
Model answer: {answer}

Output:
"""
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=20
        )
        label = response.output_text.strip()
        if label in ["0", "0.5", "1"]:
            return label
        else:
            return None
    except Exception as e:
        print("Error:", e)
        return None


# ======== Load dataset ==========
df = pd.read_csv("pilot_results.csv")

labels = []

print("Running hallucination analysis...\n")
for i, r in tqdm(df.iterrows(), total=len(df)):
    q = r["question"]
    gold = r["gold"]
    ans = r["final_answer"]

    label = judge_hallucination(q, gold, ans)

    # Retry if model returns garbage or None
    if label is None:
        time.sleep(0.3)
        label = judge_hallucination(q, gold, ans)

    labels.append(label)

    time.sleep(0.3)   # avoid rate limit

df["hallu_label"] = labels
df.to_csv("pilot_results_with_hallu.csv", index=False)

print("DONE! Saved to pilot_results_with_hallu.csv")
