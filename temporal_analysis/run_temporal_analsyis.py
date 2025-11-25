import sys
import os
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to temporal_analysis/, then into prompt_style/scripts
scripts_dir = os.path.join(os.path.dirname(current_dir), 'prompt_style', 'scripts')
sys.path.append(scripts_dir)
from pilot import load_since1500
from load_dataset import load_prehistory_qa
from test_qa_pairs import evaluate_with_fact_checking
import pandas

from dotenv import load_dotenv
from openai import OpenAI
import argparse
import pandas as pd
from datasets import load_dataset


MODEL_NAME = "gpt-4.1-mini"  # Using gpt-4o as GPT-5 is not yet available. Change to "gpt-5" when available.


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

USE_FULL = False

if USE_FULL:
    sample = df.copy().reset_index(drop=True)
    print(f"Using full dataset: {len(sample)} questions.")
else:
    N = 100
    sample = df.sample(N, random_state=42).reset_index(drop=True)
    print(f"Sampled {N} questions.")

print("Evaluating...")
results = evaluate_with_fact_checking(client, MODEL_NAME, sample)
pandas.DataFrame(results).to_csv("temporal_results.csv", index=False)

