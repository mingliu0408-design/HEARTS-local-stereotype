import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def bootstrap_macro_f1(y_true, y_pred, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        score = f1_score(y_true[idx], y_pred[idx], average="macro")
        scores.append(score)

    scores = np.array(scores)
    mean = scores.mean()
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return mean, lower, upper, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join("local_context", "zh_results", "bootstrap_macro_f1.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)

    y_true = df["true_label"].values
    y_pred = df["pred_label"].values

    mean, lower, upper, scores = bootstrap_macro_f1(
        y_true, y_pred, n_bootstrap=args.n_bootstrap
    )

    print(f"\nBootstrap Macro-F1 for {args.model_name}:")
    print(f"  mean = {mean:.4f}")
    print(f"  95% CI = [{lower:.4f}, {upper:.4f}]")

    row = {
        "model_name": args.model_name,
        "n_bootstrap": args.n_bootstrap,
        "macro_f1_mean": mean,
        "macro_f1_ci_lower": lower,
        "macro_f1_ci_upper": upper,
    }

    if os.path.exists(args.out_csv):
        out_df = pd.read_csv(args.out_csv)
        out_df = pd.concat([out_df, pd.DataFrame([row])], ignore_index=True)
    else:
        out_df = pd.DataFrame([row])

    out_df.to_csv(args.out_csv, index=False)
    print(f"\n💾 Saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
