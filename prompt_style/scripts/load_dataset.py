import pandas as pd
import json
import requests
from pathlib import Path

def load_prehistory_qa(source):
    """
    Load historical QA dataset from:
      - GitHub raw URL
      - Local JSON file
      - Path object

    Expected format:
    {
        "metadata": {...},
        "qa_pairs": [
            {"question": "...", "answer": "..."},
            ...
        ]
    }

    Returns:
        A DataFrame with columns: question, gold
    """

    # Case 1: GitHub raw url
    if isinstance(source, str) and source.startswith("http"):
        print("Loading dataset from GitHub raw URL...")
        response = requests.get(source)
        data = json.loads(response.text)

    # Case 2: Local JSON file path
    else:
        print("Loading dataset from local file...")
        path = Path(source)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Convert json structure -> DataFrame
    df = pd.DataFrame(data["qa_pairs"])
    df = df.rename(columns={"answer": "gold"})

    print(f"Loaded {len(df)} QA pairs.")
    return df
