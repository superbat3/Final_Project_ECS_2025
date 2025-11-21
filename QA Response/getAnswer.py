import os
import pandas as pd
import json
from tqdm import tqdm
from collections import Counter
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = "data/sampledQA.jsonl"
OUTPUT_PATH = "data/4omini-direct.jsonl"


def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df
# load dataset
df = load_dataset(DATA_PATH)
print(df.head(3))


# define function to call the model
def call_model(question, model_name="gpt-4o-mini", prompt_type="direct"):
    PROMPTS = {
    "direct": "Question: {question}\nAnswer the question concisely and accurately. Make sure your answer is complete but does not exceed 200 words.",
    "historian": (
        "You are a professional historian.\n"
        "Answer the following question as if you are explaining to a student.\n"
        "Question: {question}\nMake sure your answer is complete but does not exceed 200 words.\nAnswer:"
    ),
    "cot": "Question: {question}\nThink through the question step by step internally, considering relevant facts, causes, and consequences. Then provide a concise final answer. Make sure your answer is complete but does not exceed 200 words.\nAnswer:"
    }
    prompt_template = PROMPTS.get(prompt_type, PROMPTS["direct"])
    prompt = prompt_template.format(question=question)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300
    )
    return resp.choices[0].message.content.strip()



results = []
df_run = df

model_name = "gpt-4o-mini"
prompt_type = "direct"
# run the model on each question
for idx, row in tqdm(df_run.iterrows(), total=len(df_run), desc=f"{model_name}-{prompt_type}"):
    try:
        answer = call_model(row['question'], model_name=model_name, prompt_type=prompt_type)
    except Exception as e:
        print(f"Error on {row['id']}:", e)
        answer = ""
    results.append({
        "id": row['id'],
        "period": row['period'],
        "model": model_name,
        "prompt_type": prompt_type,
        "question": row['question'],
        "gold_answer": row['answer'],
        "model_answer": answer
    })

# save output
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Saved outputs to {OUTPUT_PATH}")