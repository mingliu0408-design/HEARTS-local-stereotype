## 1. Task Definition (Local Context)

### Binary classification

- `label = 1`: stereotype  
- `label = 0`: non-stereotype  
- `group = "gender"` is kept only for schema consistency  
  (not the main fairness analysis dimension).

### Metrics used (main)

- Accuracy  
- Macro-F1  

### Plus statistical tests

- McNemar test (paired significance)  
- Bootstrap 95% CI for Macro-F1


## 2. Experiment Environment

Runs performed on:

- **Device**: MacBook Air (M2, 2022)
- **Chip**: Apple M2
- **Memory**: 8 GB
- **OS**: macOS Sequoia 15.3.2


## 3. Repository Structure

Current project layout (as in the repo):

HEARTS-local-stereotype/
├── baseline/
│   ├── BERT_MGSD_baseline.py
│   ├── classification_report.csv
│   ├── full_results.csv
│   └── MGSD.csv
│
├── local_context/
│   ├── ablation_data_size.py
│   ├── baseline_tfidf_lr.py
│   ├── bootstrap_macro_f1.py
│   ├── significance_mcnemar_simple.py
│   ├── train_zh_bert.py
│   ├── train_zh_macbert.py
│   ├── preprocessing.py
│   ├── data/
│   │   ├── chinese_stereotypes.csv
│   │   ├── chinese_stereotypes_ChatGPT1000data.csv
│   │   └── processed/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       ├── test.csv
│   │       └── train_full_backup.csv
│   └── zh_results/
│       ├── ablation_data_size_agg.csv
│       ├── ablation_data_size_results.csv
│       ├── baseline_tfidf_lr_classification_report.csv
│       ├── baseline_tfidf_lr_full_results.csv
│       ├── bootstrap_macro_f1.csv
│       ├── mcnemar_results.csv
│       ├── codecarbon_*.csv
│       ├── emissions_summary_*.txt
│       └── zh_*_report*.csv / zh_full_results*.csv
│
├── figs/
├── requirements.txt
└── README.md


> **Note**: Training output folders like `zh_model_*` / `zh_model_ablate_*` can be large.  
> For GitHub submission, it is usually sufficient to keep **final CSV outputs** in  
> `local_context/zh_results/` plus figures in `figs/`.

## 4. Data (Local Context)

### 4.1 Local dataset source: CORGI-PM

We use CORGI-PM paper/dataset resources to construct a local-context Chinese dataset.

### 4.2 Balanced pool construction (5000)

`local_context/data/chinese_stereotypes.csv` is a **manually constructed balanced pool**:

- take the **first 2500** samples from `stereotype`
- take the **first 2500** samples from `non-stereotype`
- merge into a **1:1 dataset (5000 total)**

`local_context/data/chinese_stereotypes_ChatGPT1000data.csv` contains an early trial with
LLM-synthesized data (used only for analysis / discussion of limitations).

### 4.3 Preprocessing & split

Use `local_context/preprocessing.py` to:

- clean text (strip)
- map labels: `stereotype -> 1`, `non-stereotype -> 0`
- stratified split into **70 / 15 / 15**

**Outputs:**

- `local_context/data/processed/train.csv` (3500)
- `local_context/data/processed/val.csv` (750)
- `local_context/data/processed/test.csv` (750)

**Run:**

```bash
python local_context/preprocessing.py
```

## 5. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Baseline Replication (MGSD)

**Script:**

- `baseline/BERT_MGSD_baseline.py`

**Run:**

```bash
python baseline/BERT_MGSD_baseline.py
```

**Outputs (MGSD):**

- `baseline/classification_report.csv`
- `baseline/full_results.csv`

**Replication summary:**

- HEARTS reports BERT Macro-F1 ≈ **81.2%** on MGSD test.
- Our replication achieves Macro-F1 ≈ **0.8117 (81.17%)**, meeting the ±5% requirement.

## 7. Local Context Experiments (Chinese)

### 7.1 Weak baseline: TF-IDF + Logistic Regression

**Run:**

```bash
python local_context/baseline_tfidf_lr.py
```

**Outputs:**

- `local_context/zh_results/baseline_tfidf_lr_classification_report.csv`
- `local_context/zh_results/baseline_tfidf_lr_full_results.csv`

### 7.2 Main model: Chinese BERT fine-tuning

**Run (example):**

```bash
python local_context/train_zh_bert.py --epochs 3 --lr 2e-5 --run_name e3_lr2e-5
```

**Typical tuned runs tested:**

- `e3_lr2e-5` (**selected**)
- `e3_lr3e-5`
- `e5_lr2e-5`
- `e5_lr3e-5`

**Outputs (examples):**

- `local_context/zh_results/zh_classification_report_e3_lr2e-5.csv`
- `local_context/zh_results/zh_full_results_e3_lr2e-5.csv`

### 7.3 Strong baseline: MacBERT

**Run (example):**

```bash
python local_context/train_zh_macbert.py --epochs 3 --lr 2e-5 --run_name macbert_e3_lr2e-5
```

**Outputs (examples):**

- `local_context/zh_results/zh_classification_report_macbert_e3_lr2e-5.csv`
- `local_context/zh_results/zh_full_results_macbert_e3_lr2e-5.csv`


## 8. Ablation: Training Data Size (Required)

**Ablation sizes:**

- 500 / 1000 / 2000 / 3000 / 3500

**Seeds:**

- 42 / 43 / 44  

(Seed controls **subset sampling**; training randomness is kept fixed as configured in training scripts.)

**Run:**

```bash
python local_context/ablation_data_size.py \
  --sizes 500,1000,2000,3000,3500 \
  --seeds 42,43,44 \
  --epochs 3 \
  --lr 2e-5
```


### How it works (implementation detail)

- The script creates a backup of the original full training set  
  (`local_context/data/processed/train_full_backup.csv`) and temporarily
  overwrites `train.csv` with a **stratified 1:1 subset** for each `(size, seed)` run.
- After all runs, it restores the original training set.

**Outputs:**

- `local_context/zh_results/ablation_data_size_results.csv` (all runs)
- `local_context/zh_results/ablation_data_size_agg.csv` (mean ± std per size)


## 9. Statistical Significance & Confidence Intervals (Required)

### 9.1 McNemar test (paired significance)

**Run:**

```bash
python local_context/significance_mcnemar_simple.py
```

**Output:**

- `local_context/zh_results/mcnemar_results.csv`

### 9.2 Bootstrap 95% CI for Macro-F1

**Run:**

```bash
python local_context/bootstrap_macro_f1.py
```

**Output:**

- `local_context/zh_results/bootstrap_macro_f1.csv`

## 10. Failure Case Analysis (Required)

Failure cases are sourced from per-example prediction files:

- `local_context/zh_results/zh_full_results*.csv` (BERT)
- `local_context/zh_results/zh_full_results_macbert*.csv` (MacBERT)
- `local_context/zh_results/baseline_tfidf_lr_full_results.csv` (TF-IDF)

In the report, we select **10–15 representative errors** and group them into **2–3 categories**
(e.g., negation/irony, implicit stereotypes, borderline cases), and highlight cases where
**MacBERT corrects BERT**.

## 11. Key Results (Local Context Summary)

**Baseline comparison (Macro-F1):**

- TF-IDF: ~0.50
- BERT: ~0.834
- MacBERT: ~0.840 (small gain)

**Significance:**

- TF-IDF vs BERT: significant improvement (McNemar p < 0.05)
- BERT vs MacBERT: not significant (p ≥ 0.05)

**Ablation:**

- Performance improves with more data but shows diminishing returns after ~1000 samples.

(All result CSVs are in `local_context/zh_results/`.)

## 12. Sustainability / Compute Note

CodeCarbon outputs on macOS (Apple M2) may be incomplete or unreliable across runs.
We therefore discuss sustainability **qualitatively**:

- TF-IDF baseline is lightweight.
- Transformer fine-tuning is more compute-intensive.
- Ablation indicates diminishing returns → compute / performance trade-off.

## 13. (Optional) Coursework Requirement Mapping (Where to find evidence)

- **Baseline replication (±5%)**:  
  `baseline/BERT_MGSD_baseline.py` + `baseline/classification_report.csv`

- **Local dataset & split**:  
  `local_context/data/chinese_stereotypes.csv` +  
  `local_context/preprocessing.py` +  
  `local_context/data/processed/*.csv`

- **Hyperparameter tuning**:  
  `local_context/train_zh_bert.py` outputs in `local_context/zh_results/`

- **Baselines comparison**:  
  TF-IDF (`local_context/baseline_tfidf_lr.py`), BERT, MacBERT

- **Ablation**:  
  `local_context/ablation_data_size.py` +  
  `local_context/zh_results/ablation_data_size_agg.csv`

- **Significance & CI**:  
  `local_context/significance_mcnemar_simple.py`,  
  `local_context/bootstrap_macro_f1.py`

- **Failure cases**:  
  per-example `*full_results*.csv` files in `local_context/zh_results/`

## 14. References

- HEARTS paper (baseline replication reference)
- CORGI-PM paper / dataset resources (local-context data source reference)
