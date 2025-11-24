from datasets import load_dataset
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from load_dataset import load_prehistory_qa
import argparse


# ============================
# 0. 配置模型 Client
# ============================
load_dotenv() 
client = OpenAI()


# ============================
# 0. Argument Parser
# ============================
def get_args():
    parser = argparse.ArgumentParser(
        description="Generate LLM answers for historical QA dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="since1500",
        choices=["since1500", "to1500", "both"],
        help="Choose dataset: since1500 | to1500 | both"
    )

    args = parser.parse_args()
    print(f"[INFO] Using dataset = {args.dataset}")
    return args

args = get_args()


# ============================
# 1. Data Loading
# ============================
# Load since1500 (HuggingFace)
def load_since1500():
    raw = load_dataset("nielsprovos/world-history-1500-qa")
    df = raw["train"].to_pandas()

    # explode qa_pairs
    df = df.explode("qa_pairs", ignore_index=True)

    # unpack dict
    qa = pd.json_normalize(df["qa_pairs"])
    df = pd.concat([df.drop(columns=["qa_pairs"]), qa], axis=1)

    df = df[["question", "answer"]]
    df.rename(columns={"answer": "gold"}, inplace=True)

    return df


# Choose dataset based on args
print("Loading dataset...")
if args.dataset == "to1500":
    df = load_prehistory_qa("https://raw.githubusercontent.com/provos/world-history-to-1500-qa/master/dataset.json")

elif args.dataset == "since1500":
    df = load_since1500()

elif args.dataset == "both":
    df1 = load_since1500()
    df2 = load_prehistory_qa("https://raw.githubusercontent.com/provos/world-history-to-1500-qa/master/dataset.json")
    df1["period"] = "since1500"
    df2["period"] = "to1500"
    df = pd.concat([df1, df2]).reset_index(drop=True)



print("Dataset loaded. Total QA pairs:", len(df))



# ============================
# 2. 随机抽取问题
# ============================
USE_FULL = True

if USE_FULL:
    sample = df.copy().reset_index(drop=True)
    print(f"Using full dataset: {len(sample)} questions.")
else:
    N = 30
    sample = df.sample(N, random_state=42).reset_index(drop=True)
    print(f"Sampled {N} questions.")



# ============================
# 3. Prompt 模板
# ============================
PROMPT_DIRECT = """You are a knowledgeable historian.
Answer the following question with a short, direct answer in one sentence.

Question: {q}
Answer:
"""

PROMPT_EXPLAIN = """You are a knowledgeable historian.
First explain your reasoning in 2-4 sentences.
Then, on a new line starting with 'Final answer:', give the final answer.

Question: {q}
"""

PROMPT_FACT = """You are a careful historian.
Only state facts that are widely verified by historical sources.
Give a short answer in bullet points (1-3 bullets).

Question: {q}
Answer:
"""



# ============================
# 4. 模型调用函数
# ============================
def ask_model(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content



# ============================
# 5. 运行三种 prompt 得到模型回答
# ============================
rows = []

print("Running model on sampled questions...\n")

for i, r in tqdm(sample.iterrows(), total=len(sample), desc="Questions"):
    q = r["question"]
    gold = r["gold"]

    for style, template in tqdm(
        [
            ("direct", PROMPT_DIRECT),
            ("explain", PROMPT_EXPLAIN),
            ("fact", PROMPT_FACT),
        ],
        desc=f"Prompts for Q{i}",
        leave=False
    ):
        prompt = template.format(q=q)
        raw = ask_model(prompt)

        # 提取 Final answer（如果存在）
        if style == "explain" and "Final answer:" in raw:
            final = raw.split("Final answer:")[1].strip()
        else:
            final = raw.strip()

        rows.append({
            "id": i,
            "prompt_style": style,
            "question": q,
            "gold": gold,
            "raw_answer": raw,
            "final_answer": final
        })

out = pd.DataFrame(rows)
out.to_csv("pilot_results.csv", index=False)

print("Saved pilot_results.csv. You may move it to data/ manually.")
