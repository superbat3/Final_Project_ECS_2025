# LLM Factual Reliability on Historical QA
### *Prompt Sensitivity, Hallucination Behavior, and Metric-Based Evaluation*

This project evaluates the factual reliability of **Large Language Models (LLMs)** when answering historical questions. We analyze how prompt design influences accuracy, stability, and hallucination behavior using multiple evaluation metrics across two historical QA datasets. We also analyze how the time period which a question is centered on affects GPT's hallucination rate.

# Project Overview

Large Language Models (LLMs) are strong in text generation but may produce inaccurate or hallucinated historical facts.
This project systematically examines:

- **Prompt Sensitivity** – How different prompting styles influence accuracy

- **Hallucination Behavior** – Using LLM-as-a-Judge to detect factual errors

- **Metric Evaluation** – F1, BLEU, and BERTScore

- **Cross-Dataset Comparison** – To-1500 vs Since-1500 history QA

- **Full Automated Pipeline** – One command runs entire experiment

# Project Structure
```
project_root/
│
├── prompt_style/
│   ├── scripts/
│   │   ├── pilot.py
│   │   ├── eval_all.py
│   │   ├── plot_results.py
│   │   └── load_dataset.py
│   ├── data/        ← CSV outputs placed here manually
│   └── results/     ← generated plots placed here manually
│
├── hallucination_study/
│   ├── scripts/
│   │   ├── hallucination_judge.py
│   │   ├── eval_hallucination.py
│   │   └── plot_hallucination.py
│   ├── data/        ← CSV outputs placed here manually
│   └── results/     ← generated plots placed here manually
│
├── temporal_analysis/
│   ├── visualize_correlations.py
│   ├── visualize_conditional.py
│   ├── calculate_question_difficulty.py
│   ├── run_temporal_analsyis.py
│   ├── test_qa_pairs.py
│   ├── finalsince.csv / finalto.csv           ← consolidated datasets for temporal analysis
│   └── plots/                                ← generated plots (correlations_plot.png, conditional_and_bins_plot.png)
│
├── run_all.py       ← Full pipeline runner
├── requirements.txt
└── README.md
```

# Datasets
### 1. World History Since 1500

Loaded using HuggingFace:

`dataset = load_dataset("nielsprovos/world-history-1500-qa")`

### 2. World History to 1500

Raw JSON from GitHub:

`https://raw.githubusercontent.com/provos/world-history-to-1500-qa/master/dataset.json`


Each dataset contains:

`question` and `answer` (renamed as `gold`)

# Prompt Styles
1. **Direct Prompt**
```
Answer the following historical question concisely.
Question: {q}
```
2. **Explain Prompt**
```
Explain your reasoning step-by-step, and provide a final answer.
Question: {q}
```
3. **Fact-only Prompt**
```
Give a short answer in 1–3 bullet points.
Question: {q}
```

# Evaluation Metrics
Quantitative Metrics

| Metric | Description |
|-----|------|
| F1   | Token-level precision & recall    |
| BLEU   | N-gram similarity    |
| BERTScore   | Semantic similarity    |

Qualitative Metrics

| Metric | Description |
|-----|------|
| Hallucination Label (0 / 0.5 / 1)   | uman-like/LLM-as-judge evaluation   |
| Hallucination rate   | How often hallucination appears    |
| Severity distribution   | Proportion of 0 / 0.5 / 1    |

# Hallucination Detection (LLM-as-a-Judge)

We use a judge LLM to evaluate:

- factual correctness

- unsupported details

- fabricated events or names

Labels:

- 0 = Correct

- 0.5 = Slight hallucination

- 1 = Major hallucination

Judge runs automatically through:

`hallucination/scripts/hallucination_judge.py`


# Full Pipeline (One Command)
```
python run_all.py --dataset since1500
```

Options:

    --dataset since1500
    --dataset to1500
    --dataset both


Pipeline includes:

- Generate responses

- Compute F1/BLEU/BERTScore

- Hallucination analysis

- Generate all plots

# Installation
    git clone <your_repo>
    cd project_root

    pip install -r requirements.txt


Ensure .env contains:

    OPENAI_API_KEY="your_key"


# Acknowledgments

- UC Davis ECS 289A

- HuggingFace Team

- OpenAI GPT Models