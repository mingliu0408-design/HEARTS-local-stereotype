import argparse
import os
import pandas as pd
from math import fabs
from scipy.stats import chi2


def load_preds(path: str):
    df = pd.read_csv(path)
    return df["true_label"].values, df["pred_label"].values


def run_mcnemar(y_true_a, y_pred_a, y_true_b, y_pred_b):
    assert (y_true_a == y_true_b).all(), "Test sets are not identical!"

    a_correct = y_pred_a == y_true_a
    b_correct = y_pred_b == y_true_b

    # n01: A wrong, B correct
    # n10: A correct, B wrong
    n01 = ((~a_correct) & (b_correct)).sum()
    n10 = ((a_correct) & (~b_correct)).sum()

    # McNemar chi-square with continuity correction
    # (|n01 - n10| - 1)^2 / (n01 + n10)
    if (n01 + n10) == 0:
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (fabs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return int(n01), int(n10), float(chi2_stat), float(p_value)


def append_to_csv(out_csv: str, row: dict):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--name_a", type=str, default="ModelA")
    parser.add_argument("--name_b", type=str, default="ModelB")
    parser.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join("local_context", "zh_results", "mcnemar_results.csv"),
        help="Where to save results (will append).",
    )
    args = parser.parse_args()

    y_true_a, y_pred_a = load_preds(args.model_a)
    y_true_b, y_pred_b = load_preds(args.model_b)

    n01, n10, stat, pval = run_mcnemar(y_true_a, y_pred_a, y_true_b, y_pred_b)

    # Console output (keep it readable)
    print(f"\nMcNemar contingency table ({args.name_a} vs {args.name_b}):")
    print(f"  {args.name_a} wrong, {args.name_b} correct: {n01}")
    print(f"  {args.name_a} correct, {args.name_b} wrong: {n10}")

    print("\nMcNemar test result:")
    print(f"  statistic = {stat:.4f}")
    print(f"  p-value   = {pval:.6f}")

    sig = pval < 0.05
    if sig:
        print("  ✅ Difference is statistically significant (p < 0.05)")
    else:
        print("  ❌ Difference is NOT statistically significant (p ≥ 0.05)")

    # Save to CSV
    row = {
        "model_a_name": args.name_a,
        "model_b_name": args.name_b,
        "model_a_path": args.model_a,
        "model_b_path": args.model_b,
        "n_a_wrong_b_correct": n01,
        "n_a_correct_b_wrong": n10,
        "statistic": stat,
        "p_value": pval,
        "significant_p_lt_0_05": sig,
    }
    append_to_csv(args.out_csv, row)
    print(f"\n💾 Saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
