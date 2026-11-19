# Home Credit Default Risk — GCI Global 2026 Competition Submission

This repository contains my submitted solution for the **In-class Competition: Home Credit Default Risk** in **GCI Global 2026**, organized by Matsuo-Iwasawa Laboratory, University of Tokyo.

## Competition summary

| Item | Details |
|---|---|
| Competition | In-class Competition: Home Credit Default Risk |
| Course | GCI Global 2026 |
| Program schedule | September 17 – December 17, 2026 |
| Format | 3-month online program in English |
| Organizer | Matsuo-Iwasawa Laboratory, University of Tokyo |
| Objective | Predict the probability that a customer will experience payment difficulties |
| Target | `1` = payment difficulties; `0` = no payment difficulties |
| Evaluation metric | ROC-AUC |
| Submission file | `output/submission.csv` |
| Public leaderboard score | **0.763848** |

## Brief results

- **Final approach:** 5-fold CatBoost classifier with engineered application features.
- **5-fold out-of-fold ROC-AUC:** `0.760103`.
- **Random Forest holdout ROC-AUC:** `0.734796` for comparison.
- **Public leaderboard ROC-AUC:** **`0.763848`**.
- **Final submission:** `output/submission.csv`, containing `SK_ID_CURR` and `TARGET` for 61,500 test records.
- **Final prediction method:** Average the predicted probabilities from all five CatBoost folds.

## Notebook objective

[tutorial.ipynb](tutorial.ipynb) is my end-to-end notebook for generating default-risk probabilities. It loads the training and test application data, explores the data, creates additional features, validates CatBoost with stratified 5-fold cross-validation, compares it with a Random Forest baseline, and writes the required competition submission.

## End-to-end workflow

1. Import the Python libraries.
2. Mount Google Drive when using Google Colab.
3. Set the working directory and verify that the data files are available.
4. Load `train.csv`, `test.csv`, and `submission.csv`.
5. Inspect target balance, feature types, distributions, missing values, and outliers.
6. Split `train.csv` into `TARGET` and explanatory features.
7. Engineer age, employment, financial-ratio, external-source, and bureau-enquiry features.
8. Identify categorical features and prepare them for CatBoost.
9. Train CatBoost with 5-fold stratified cross-validation.
10. Compare CatBoost with a Random Forest holdout experiment.
11. Use the averaged 5-fold CatBoost test probabilities for the final submission.
12. Save and verify `output/submission.csv`.

## Data files

The notebook expects the competition files in the active working directory. A Colab setup is included, and it checks files under `/content`.

| File | Purpose |
|---|---|
| `train.csv` | 171,202 labeled application records with `SK_ID_CURR` and `TARGET` |
| `test.csv` | 61,500 application records for probability prediction; no `TARGET` column |
| `submission.csv` | Competition submission template or sample file; used to locate the submission format |
| `HomeCredit_columns_description.xlsx` | Optional reference for understanding application fields |
| `output/submission.csv` | Final generated submission produced by the notebook |

The original notebook output reports 34 columns in `train.csv` and 33 columns in `test.csv`. These counts describe the competition subset used for this tutorial.

## Preprocessing and feature engineering

- Replaces the sentinel value `365243` in `DAYS_EMPLOYED` with missing values.
- Converts categorical columns to strings and fills missing categorical values with `"Missing"` for CatBoost.
- Leaves CatBoost to handle numerical missing values in the current pipeline.
- Identifies categorical features automatically using pandas object/category dtypes.
- For the Random Forest comparison only:
  - one-hot encodes categorical features with `pd.get_dummies`;
  - aligns training and validation columns;
  - fills missing numerical values with `0`.

The notebook creates:

- `AGE_YEARS`
- `EMPLOYMENT_YEARS`
- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_ANNUITY_RATIO`
- `GOODS_CREDIT_RATIO`
- `INCOME_PER_FAMILY`
- `EXT_SOURCE_MEAN`
- `EXT_SOURCE_MIN`
- `EXT_SOURCE_MAX`
- `EXT_SOURCE_STD`
- `TOTAL_BUREAU_ENQUIRIES`

## Model development and validation

### Primary model: CatBoost

The final model is an averaged ensemble of five CatBoost classifiers trained with:

- 5-fold stratified shuffled cross-validation
- Random seed `42 + fold`
- 1,000 maximum iterations
- Learning rate `0.05`
- Depth `6`
- `Logloss` loss
- `AUC` evaluation metric
- `l2_leaf_reg=5`
- 150-round early stopping

Each fold predicts the validation partition and contributes one fifth of the test probability prediction.

### Validation results recorded in the notebook

| Result | Value |
|---|---:|
| CatBoost 5-fold out-of-fold ROC-AUC | 0.760103 |
| Random Forest holdout ROC-AUC | 0.734796 |
| Weighted ensemble validation ROC-AUC (70% CatBoost + 30% Random Forest) | 0.758284 |
| Public leaderboard score | 0.763848 |

The notebook also recorded the Random Forest training ROC-AUC as `0.829071`.

The solution uses CatBoost test predictions for the final submission because the recorded validation result is stronger than the Random Forest comparison and the weighted ensemble.

Areas that remain for further tuning are hyperparameters, feature selection, alternative imbalance strategies, and additional model or ensemble experiments. These are suggestions in the notebook, not claims of an implemented improvement.

## How to run

### Google Colab

1. Open [tutorial.ipynb](tutorial.ipynb) in Google Colab.
2. Mount Google Drive when prompted.
3. Upload or link `train.csv`, `test.csv`, and `submission.csv` in `/content`, or update the working-directory cell to the folder containing the notebook and data.
4. Run all cells from Section 2.1 onward after modifying data or features.
5. Confirm that `output/submission.csv` is created.

### Local Jupyter

```bash
jupyter notebook tutorial.ipynb
```

Place the competition CSV files in the same directory as the notebook, or adjust `train_file`, `test_file`, and `sample_sub_file` in the path setup cell.

```python
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
sample_sub = pd.read_csv(sample_sub_file)
```

Install CatBoost if it is not already available:

```bash
pip install catboost
```

## Submission format

The submission must contain exactly these required columns:

| Column | Description |
|---|---|
| `SK_ID_CURR` | Test application identifier |
| `TARGET` | Predicted probability of default/payment difficulties |

Expected shape:

```text
61,500 rows
SK_ID_CURR, TARGET
```

The notebook creates the file with:

```python
submission = pd.DataFrame({
    "SK_ID_CURR": test_ids,
    "TARGET": pred
})
submission.to_csv("output/submission.csv", index=False)
```

The final probabilities are the mean of five CatBoost fold predictions. The code verifies that the saved file has 61,500 records and prints its first and last rows.

## Reproducibility

- CatBoost uses `random_seed=42 + fold`.
- Stratified cross-validation uses seed `42`.
- The Random Forest comparison uses `random_state=0`.
- Run notebook cells in order and reload the data after changing preprocessing or feature creation.
- The notebook suppresses warnings and disables CatBoost file writing.
- Package versions and a machine-specific environment can affect exact binary outputs, so record package versions in the submitted environment if exact reruns are required.


## Sources and references

- [Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk) — competition description and evaluation context.
