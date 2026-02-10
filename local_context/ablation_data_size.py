import os
import argparse
import subprocess
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="500,1000,2000,3000,5000")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    train_path = os.path.join("local_context", "data", "processed", "train.csv")
    backup_path = os.path.join("local_context", "data", "processed", "train_full_backup.csv")

    # backup full train once
    if not os.path.exists(backup_path):
        pd.read_csv(train_path).to_csv(backup_path, index=False)
        print("✅ Backed up full train to:", backup_path)

    full_train = pd.read_csv(backup_path)

    out_rows = []

    for size in sizes:
        # stratified sample (keep class balance)
        # assumes label column exists and is 0/1
        df0 = full_train[full_train["label"] == 0]
        df1 = full_train[full_train["label"] == 1]
        n0 = size // 2
        n1 = size - n0
        if len(df0) < n0 or len(df1) < n1:
            raise ValueError(f"Not enough samples for size={size}. Have: label0={len(df0)}, label1={len(df1)}")

        for seed in seeds:
            sub0 = df0.sample(n=n0, random_state=seed)
            sub1 = df1.sample(n=n1, random_state=seed)
            sub_train = pd.concat([sub0, sub1]).sample(frac=1, random_state=seed).reset_index(drop=True)

            # overwrite train.csv with subset
            sub_train.to_csv(train_path, index=False)

            run_name = f"ablate_size{size}_seed{seed}_e{args.epochs}_lr{args.lr}"
            cmd = [
                "python", os.path.join("local_context", "train_zh_bert.py"),
                "--epochs", str(args.epochs),
                "--lr", str(args.lr),
                "--run_name", run_name
            ]
            print("\n🚀 Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            # read report produced by your train_zh_bert.py
            report_file = os.path.join("local_context", "zh_results", f"zh_classification_report_{run_name}.csv")
            rep = pd.read_csv(report_file, index_col=0)

            macro_f1 = float(rep.loc["macro avg", "f1-score"])
            acc = float(rep.loc["accuracy", "precision"])  # sklearn puts accuracy in that column

            out_rows.append({
                "train_size": size,
                "seed": seed,
                "epochs": args.epochs,
                "lr": args.lr,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "report_file": report_file
            })

    # restore original train.csv
    pd.read_csv(backup_path).to_csv(train_path, index=False)
    print("\n🔄 Restored full train.csv")

    out_df = pd.DataFrame(out_rows)
    out_path = os.path.join("local_context", "zh_results", "ablation_data_size_results.csv")
    out_df.to_csv(out_path, index=False)
    print("✅ Saved ablation summary to:", out_path)

    # also save aggregated mean/std
    agg = out_df.groupby("train_size")[["accuracy", "macro_f1"]].agg(["mean", "std"]).reset_index()
    agg_path = os.path.join("local_context", "zh_results", "ablation_data_size_agg.csv")
    agg.to_csv(agg_path, index=False)
    print("✅ Saved ablation aggregated stats to:", agg_path)

if __name__ == "__main__":
    main()
