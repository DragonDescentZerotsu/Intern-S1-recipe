## Overview

Our project is based on a **Therapeutics Data Commons (TDC)** binary classification task. Following the official TDC recommendation for many ADMET and toxicity classification datasets (e.g., *Bioavailability_Ma*, *hERG*, *AMES*, *ClinTox*), we use **Area Under the Receiver Operating Characteristic Curve (AUROC)** as our primary evaluation metric.  

AUROC is a **threshold-independent** performance measure for binary classifiers, which is particularly suitable for **imbalanced datasets**, a common setting in drug discovery and ADMET prediction.  

---

## Task and Notation

We consider a standard **binary classification** problem:

- Each example $i$ has:
  - a **ground-truth label** $y_i \in \{0, 1\}$, where  
    - $y_i = 1$ denotes the **positive class** (e.g., “toxic”, “active”, “bioavailable”),  
    - $y_i = 0$ denotes the **negative class** (e.g., “non-toxic”, “inactive”, “not bioavailable”);
  - a **model score** $s_i \in \mathbb{R}$, where larger $s_i$ means the model believes the example is **more likely positive**.

We define the index sets:

- $P = \{ i : y_i = 1 \}$ (all positive examples),  
- $N = \{ i : y_i = 0 \}$ (all negative examples),

with $|P| = m$ positives and $|N| = n$ negatives.

---

## ROC Curve

The **Receiver Operating Characteristic (ROC) curve** is a parametric curve that plots:

- **True Positive Rate (TPR)** vs. **False Positive Rate (FPR)**  
for all possible decision thresholds $\tau$ applied to the scores $s_i$.   

Given a threshold $\tau$, the classifier predicts:

$
\hat{y}_i(\tau) =
\begin{cases}
1, & \text{if } s_i \ge \tau \\
0, & \text{otherwise}
\end{cases}
$

From this we define:

- True positives:  $\text{TP}(\tau) = \left| \{ i : y_i = 1, \hat{y}_i(\tau) = 1 \} \right|$
- False positives: $\text{FP}(\tau) = \left| \{ i : y_i = 0, \hat{y}_i(\tau) = 1 \} \right|$
- True negatives:  $\text{TN}(\tau) = \left| \{ i : y_i = 0, \hat{y}_i(\tau) = 0 \} \right|$
- False negatives: $\text{FN}(\tau) = \left| \{ i : y_i = 1, \hat{y}_i(\tau) = 0 \} \right|$

Then:

$
\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}
$

$
\text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}
$

The ROC curve is the set of points:

$
\bigl( \text{FPR}(\tau), \ \text{TPR}(\tau) \bigr)
$

as \($\tau$\) ranges over all possible thresholds.

---

## Formal Definition of AUROC

### 1. Geometric (Area-Under-Curve) Definition

The **AUROC** is the area under the ROC curve in the unit square \([0,1] \times [0,1]\):   

$
\text{AUROC} = \int_{0}^{1} \text{TPR}\bigl(\text{FPR}^{-1}(u)\bigr) \, du
$

In practice, the ROC curve is approximated by a finite set of points sorted by increasing FPR, and the integral is estimated by the **trapezoidal rule**:

1. Sort all unique thresholds so that the corresponding FPR values satisfy  
   $0 = \text{FPR}_0 < \text{FPR}_1 < \dots < \text{FPR}_K = 1$.  
   Let $\text{TPR}_k$ be the TPR at the same threshold.
2. Approximate AUROC as:

$
\text{AUROC} \approx \sum_{k=0}^{K-1}
\left( \text{FPR}_{k+1} - \text{FPR}_{k} \right)
\cdot
\frac{\text{TPR}_{k+1} + \text{TPR}_{k}}{2}
$

This is equivalent to the standard implementation in common libraries such as scikit-learn.   

---

### 2. Probabilistic / Ranking Interpretation

An equivalent and often more intuitive definition is: **AUROC is the probability that a randomly chosen positive example receives a higher score than a randomly chosen negative example**.   

Formally, let $i \in P$ index positives and $j \in N$ index negatives. Then:

$
\text{AUROC}
=
\frac{1}{mn}
\sum_{i \in P}
\sum_{j \in N}
\Bigl[
\mathbf{1}(s_i > s_j) + \frac{1}{2} \mathbf{1}(s_i = s_j)
\Bigr]
$

where $\mathbf{1}(\cdot)$ is the indicator function.

This formulation makes explicit that AUROC measures how well the model **ranks** positives above negatives, independent of any particular decision threshold.

---

## Why AUROC for TDC Classification Tasks?

Therapeutics Data Commons recommends **AUROC** for many binary classification datasets in ADMET and toxicity (e.g., *Bioavailability_Ma*, *BBB_Martins*, *hERG*, *AMES*, *ClinTox*, *Tox21*).   

Key reasons this is appropriate for our project:

- Many tasks are **class-imbalanced** (e.g., relatively few toxic compounds compared to non-toxic compounds); AUROC is less sensitive to class imbalance than raw accuracy.   
- AUROC evaluates performance across **all possible thresholds**, which is useful when the appropriate decision threshold may depend on downstream application (e.g., safety vs. hit-rate trade-offs).
- AUROC is **standard in the TDC ecosystem**, enabling direct comparison with published baselines and leaderboard entries. 
