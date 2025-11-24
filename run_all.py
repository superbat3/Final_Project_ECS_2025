import argparse
import subprocess
import sys
import shutil
import os


def run(cmd):
    print("\n" + "="*60)
    print("Running:", cmd)
    print("="*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error running: {cmd}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="since1500",
                        choices=["since1500", "to1500", "both"])
    args = parser.parse_args()

    print("\nStarting FULL PIPELINE")
    print("Dataset =", args.dataset)
    print("========================================")

    # ---------- Prompt-style pipeline ----------
    print("\nPrompt-style pipeline\n")
    run(f"python prompt_style/scripts/pilot.py --dataset {args.dataset}")
    run("python prompt_style/scripts/eval_all.py")
    run("python prompt_style/scripts/plot_results.py")

    # ---------- Hallucination pipeline ----------
    print("\nHallucination pipeline\n")
    run("python hallucination_study/scripts/hallucination_judge.py")
    run("python hallucination_study/scripts/eval_hallucination.py")
    run("python hallucination_study/scripts/plot_hallucination.py")

    print("\nAll tasks completed successfully!")
    print("• Prompt results saved in prompt_style/results/")
    print("• Hallucination results saved in hallucination_study/results/")

if __name__ == "__main__":
    main()
