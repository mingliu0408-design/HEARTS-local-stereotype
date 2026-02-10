# local_context/baseline_tfidf_lr.py
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_split(split_name: str) -> pd.DataFrame:
    path = os.path.join("local_context", "data", "processed", f"{split_name}.csv")
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    if "group" not in df.columns:
        df["group"] = "gender"  # fallback, keep schema stable
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_max", type=int, default=2)
    args = parser.parse_args()

    results_dir = os.path.join("local_context", "zh_results")
    os.makedirs(results_dir, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    # Model: TF-IDF + Logistic Regression (strong, standard baseline in HEARTS-style setups)
    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=args.max_features,
                ngram_range=(1, args.ngram_max),
                lowercase=False  # Chinese: keep as-is
            )),
            ("lr", LogisticRegression(
                C=args.C,
                solver="liblinear",
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )),
        ]
    )

    print("🚀 Training TF-IDF + LR baseline...")
    clf.fit(train_df["text"], train_df["label"])

    # Optional: you could use val for tuning later; for now we just report test.
    print("📊 Evaluating baseline on test set...")
    y_true = test_df["label"].values
    y_pred = clf.predict(test_df["text"])

    # classification report (same style as your BERT script)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(results_dir, "baseline_tfidf_lr_classification_report.csv")
    df_report.to_csv(report_path)
    print("✅ Saved:", report_path)
    print(df_report)

    # detailed predictions (same columns as zh_full_results.csv)
    results_df = pd.DataFrame({
        "text": test_df["text"],
        "true_label": y_true,
        "pred_label": y_pred,
        "group": test_df["group"],
    })
    full_path = os.path.join(results_dir, "baseline_tfidf_lr_full_results.csv")
    results_df.to_csv(full_path, index=False)
    print("✅ Saved:", full_path)


if __name__ == "__main__":
    main()
