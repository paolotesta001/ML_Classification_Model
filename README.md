# Anuran Species Classification with Logistic Regression

A multiclass classification project that identifies frog (anuran) species from acoustic features extracted from their calls, using Scikit-learn.

## Dataset

The dataset comes from the [Anuran Calls (MFCCs)](https://archive.ics.uci.edu/ml/datasets/Anuran+Calls+%28MFCCs%29) collection. It contains **7195 syllables** segmented from 60 audio recordings of frogs, captured in real-world noisy environments across Brazil and Argentina.

Each instance has **22 Mel-Frequency Cepstral Coefficients (MFCCs)** as features, normalized between -1 and 1. The dataset includes three hierarchical label columns: Family (4 classes), Genus (8 classes), and Species (10 classes). This project focuses on **Species-level classification** across 10 classes:

| Species | Samples |
|---|---|
| AdenomeraAndre | 672 |
| AdenomeraHylaedactylus | 3478 |
| Ameeregatrivittata | 542 |
| HylaMinuta | 310 |
| HypsiboasCinerascens | 472 |
| HypsiboasCordobae | 1121 |
| LeptodactylusFuscus | 270 |
| OsteocephalusOophagus | 114 |
| Rhinellagranulosa | 68 |
| ScinaxRuber | 148 |

The dataset is **imbalanced** — AdenomeraHylaedactylus alone accounts for nearly half the data, while Rhinellagranulosa has only 68 samples.

## Approach

### Label Encoding

Categorical labels (Family, Genus, Species) are mapped to integer values via manual dictionaries rather than using `LabelEncoder`. This keeps the encoding explicit, reproducible, and independent of the data ordering.

### Data Split: 60/20/20

The data is split into **training (60%)**, **validation (20%)**, and **test (20%)** sets using stratified sampling (`stratify=y`). The three-way split was chosen to:

- **Train** the model on 60% of the data
- **Validate** on a held-out set to monitor overfitting via the loss curve during training
- **Test** on a completely unseen set for final, unbiased performance evaluation

Stratification ensures that each split preserves the original class distribution, which is critical given the imbalanced nature of the dataset.

### Model: SGDClassifier with Log Loss

The final model is an `SGDClassifier` with `loss='log_loss'`, which implements **logistic regression optimized via Stochastic Gradient Descent**. This was chosen over the standard `LogisticRegression` with `lbfgs` solver for a key reason:

- **Epoch-by-epoch training control**: using `warm_start=True`, the model is trained incrementally over 5000 epochs. At each epoch, both train and validation log loss are recorded. This allows plotting the **learning curve** to visually verify convergence and detect overfitting — something not easily done with `lbfgs`, which runs its optimization internally without exposing per-iteration losses.

Key hyperparameters:
- `learning_rate='constant'`, `eta0=0.01`: a fixed learning rate for stable, interpretable convergence behavior
- `max_iter=50000`: a high inner iteration cap to ensure the solver converges at each `fit()` call
- `random_state=42`: reproducibility

### Why Not Other Models?

Logistic regression was chosen as a deliberate baseline. It is a linear model well-suited for understanding the separability of MFCC features without introducing the complexity of non-linear models. The focus is on proper evaluation methodology (train/val/test split, loss curves, multiclass metrics) rather than maximizing accuracy through model complexity.

## Evaluation

The project evaluates the model using multiple complementary metrics:

### Training and Validation Loss Curve

The train and validation log loss are plotted across epochs to verify that:
- The model converges (both losses decrease and stabilize)
- There is no significant overfitting (validation loss does not diverge from training loss)

### Accuracy

Accuracy is computed on all three splits (train, validation, test) to give an overall correctness measure and to check for overfitting across splits.

### Confusion Matrix

A heatmap of the confusion matrix on the test set shows exactly which species are being confused with each other. This is especially informative for an imbalanced dataset where accuracy alone can be misleading.

### Classification Report (Precision, Recall, F1-Score)

Per-class precision, recall, and F1-score reveal how the model performs on each species individually. This is essential because aggregate accuracy can hide poor performance on minority classes like Rhinellagranulosa (68 samples) or OsteocephalusOophagus (114 samples).

### Specificity

A custom specificity function computes **True Negative Rate** for each class (TN / (TN + FP)). This complements recall (sensitivity) by showing how well the model avoids false positives for each species.

### ROC Curves and AUC (One-vs-All)

Multiclass ROC curves are plotted using a One-vs-All strategy, with AUC computed per class and as a macro-average. This provides a threshold-independent evaluation of the model's discriminative ability for each species.

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python "anuran Scikit-learn.py"
```

Make sure `anuran_calls.csv` is in the same directory as the script.


