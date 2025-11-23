from datasets import load_dataset
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from load_dataset import load_prehistory_qa

load_dotenv() 

# ============================
# 0. 配置模型 Client
# ============================
client = OpenAI()



# ============================
# 1. Data Loading
# ============================
print("Loading dataset...")

# 1. 加载
dataset = load_dataset("nielsprovos/world-history-1500-qa")

# 2. 把 train 切成 pandas
df1 = dataset["train"].to_pandas()

# 3. 正确展开：先 explode，再 json_normalize
df1 = df1.explode("qa_pairs", ignore_index=True)

# 4. qa_pairs 是一个 dict，拆开
qa = pd.json_normalize(df1["qa_pairs"])

# 5. 合并
df1 = pd.concat([df1.drop(columns=["qa_pairs"]), qa], axis=1)

# 6. 只保留 question 和 answer
df1 = df1[["question", "answer"]]
df1.rename(columns={"answer": "gold"}, inplace=True)

# print(df.head(20))
# print(df.shape)
print("Dataset 1 loaded. Total QA pairs:", len(df1))



url = "https://raw.githubusercontent.com/provos/world-history-to-1500-qa/master/dataset.json"

df2 = load_prehistory_qa(url)


# Choose one of the datasets
df = df2

# ============================
# 2. 随机抽取问题
# ============================
N = 30
sample = df.sample(N, random_state=42).reset_index(drop=True)

print(f"Sampled {N} questions.")

# sample = df.sample(3, random_state=42)
# for _, row in sample.iterrows():
#     print("Q:", row["question"])
#     print("A:", row["answer"])
#     print()



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
Give a short answer in bullet points (1\-3 bullets).

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

print("\nSaved pilot_results.csv")
print("Experiment complete!")
