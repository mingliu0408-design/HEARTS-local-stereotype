import os
import logging
import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    balanced_accuracy_score,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from codecarbon import EmissionsTracker


# -----------------------------
# Command-line arguments：epochs / lr / run_name
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--run_name", type=str, default="")
args = parser.parse_args()

# -----------------------------
# Logging setup (optional)
# -----------------------------
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)


# -----------------------------
# 1. Load processed data
# -----------------------------
def load_split(split_name: str) -> pd.DataFrame:
    """
    Load a data split from local_context/data/processed/{split_name}.csv.
    The file must contain at least text, label, and group columns.
    """
    path = os.path.join("local_context", "data", "processed", f"{split_name}.csv")
    df = pd.read_csv(path)
    print(f"✨ Loaded {split_name}.csv, total samples: {len(df)}")
    print(df.head())
    return df


# -----------------------------
# 2. Training + validation + testing + emissions tracking
# -----------------------------
def main():
    # Fix random seed for reproducibility
    set_seed(42)

    # Build suffix for file naming to avoid overwriting outputs
    run_suffix = f"_{args.run_name}" if args.run_name != "" else ""

    # Path settings
    model_name = "hfl/chinese-macbert-base"
    model_output_dir = os.path.join("local_context", f"zh_model{run_suffix}")
    results_output_dir = os.path.join("local_context", "zh_results")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    # -----------------------------
    # CodeCarbon emissions tracker
    # -----------------------------
    tracker = EmissionsTracker(
        output_dir=results_output_dir,
        output_file=f"codecarbon_zh_bert{run_suffix}.csv",
        gpu_ids=None,
        tracking_mode="process",
        measure_power_secs=1,
    )
    tracker.start()
    print("🌍 CodeCarbon EmissionsTracker started to estimate CO₂ emissions for this run.")

    try:
        # Load data
        train_df = load_split("train")
        val_df = load_split("val")
        test_df = load_split("test")

        # Ensure labels are integers
        for df in (train_df, val_df, test_df):
            df["label"] = df["label"].astype(int)

        # Convert to HuggingFace Dataset (do not keep pandas index)
        train_ds = Dataset.from_pandas(train_df, preserve_index=False)
        val_ds = Dataset.from_pandas(val_df, preserve_index=False)
        test_ds = Dataset.from_pandas(test_df, preserve_index=False)

        # Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
        )

        # Tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128,
            )

        # Apply tokenization to all splits
        train_tokenized = train_ds.map(tokenize_function, batched=True)
        val_tokenized = val_ds.map(tokenize_function, batched=True)
        test_tokenized = test_ds.map(tokenize_function, batched=True)

        # Rename label column to labels (required by Trainer)
        train_tokenized = train_tokenized.rename_column("label", "labels")
        val_tokenized = val_tokenized.rename_column("label", "labels")
        test_tokenized = test_tokenized.rename_column("label", "labels")

        # Keep only model-required columns
        def keep_only_model_columns(dataset: Dataset) -> Dataset:
            keep_cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]
            cols_to_remove = [c for c in dataset.column_names if c not in keep_cols]
            return dataset.remove_columns(cols_to_remove)

        train_tokenized = keep_only_model_columns(train_tokenized)
        val_tokenized = keep_only_model_columns(val_tokenized)
        test_tokenized = keep_only_model_columns(test_tokenized)

        # Metrics computation (for validation)
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            preds = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro"
            )
            balanced_acc = balanced_accuracy_score(labels, preds)
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_acc,
            }

        # Training arguments
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=args.lr,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            logging_steps=10,
            report_to="none",   # Do not report to external tools like wandb
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # -----------------------------
        #  Training
        # -----------------------------
        print("🚀 Starting training of Chinese BERT model...")
        trainer.train()
        print("✅ Training finished. Model saved to：", model_output_dir)
        trainer.save_model(model_output_dir)

        # -----------------------------
        # Evaluation on test set
        # -----------------------------
        print("📊 Evaluating model on the test set...")

        preds_output = trainer.predict(test_tokenized)
        logits = preds_output.predictions
        y_pred = np.argmax(logits, axis=1)
        y_true = np.array(test_df["label"].tolist())

        # classification report
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()
        report_path = os.path.join(
            results_output_dir, f"zh_classification_report{run_suffix}.csv"
        )
        df_report.to_csv(report_path)
        print("✅ Test set classification_report saved to：", report_path)
        print(df_report)

        # Save detailed results (text + predicted labels)
        results_df = pd.DataFrame(
            {
                "text": test_df["text"],
                "true_label": y_true,
                "pred_label": y_pred,
                "group": test_df["group"],
            }
        )
        full_results_path = os.path.join(
            results_output_dir, f"zh_full_results{run_suffix}.csv"
        )
        results_df.to_csv(full_results_path, index=False)
        print("✅ Detailed test predictions saved to：", full_results_path)

    finally:
        # -----------------------------
        # Stop CodeCarbon tracker
        # -----------------------------
        emissions = tracker.stop()

        # On some platforms (e.g., certain macOS + M2 setups),
        # CodeCarbon may return None. We record this as "N/A"
        if emissions is None:
            emissions_str = "N/A"
            print("🌍 CodeCarbon did not return an emissions estimate (startup failure or unsupported environment)。")
        else:
            emissions_str = f"{emissions:.6f}"
            print(f"🌍 CodeCarbon estimated emissions for this run：{emissions_str} kg CO₂")

        # Write a short text summary for easy reference in README or poster
        summary_path = os.path.join(
            results_output_dir, f"emissions_summary{run_suffix}.txt"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Estimated emissions for train_zh_bert.py run:\n")
            f.write(f"run_name = {args.run_name}\n")
            f.write(f"epochs = {args.epochs}, lr = {args.lr}\n")
            f.write(f"emissions = {emissions_str} kg CO2\n")
        print("📝 Emissions summary written to：", summary_path)


if __name__ == "__main__":
    main()
