import pandas as pd

df = pd.read_csv("pilot_results_with_hallu.csv")

# 平均 hallucination rate
rate = df.groupby("prompt_style")["hallu_label"].mean()

# 严重幻觉比例（label==1）
severe = (df["hallu_label"] == 1).groupby(df["prompt_style"]).mean()

summary = pd.DataFrame({
    "hallucination_rate": rate,
    "severe_rate": severe
})

print(summary)
summary.to_csv("hallucination_summary.csv")
print("Saved hallucination_summary.csv")
