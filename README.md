# Double Descent in Polynomial Regression

This notebook explores the double descent phenomenon using synthetic sinusoidal data and real-world weather data, with Ridge and Lasso regularization.

View the notebook: [Math156_FinalProject.ipynb](DoubleDescent.ipynb)  
Final report: [Math156_FinalPaper.pdf](DoubleDescentReport.pdf)

## Project Overview

Classical learning theory predicts a **U-shaped bias–variance tradeoff**: test error decreases with model complexity, then increases as the model overfits. Double descent describes a more modern behavior where test error:

1. **Decreases** as model complexity increases (underparameterized regime)  
2. **Spikes** near the interpolation threshold (model just complex enough to fit training data)  
3. **Decreases again** in the overparameterized regime  

In this project, we:

- Recreate double descent using **polynomial regression** on a noisy sine wave
- Introduce **L1 (Lasso)** and **L2 (Ridge)** regularization
- Apply the same pipeline to **real daily maximum temperatures** from Albury, Australia
- Compare how each regression method behaves across model complexities, both in terms of **MSE** and **solution curves** :contentReference[oaicite:1]{index=1}  

---

## Repository Contents

- `DoubleDescent.ipynb` – main analysis notebook (synthetic + weather experiments)
- `DoubleDescentReport.pdf` – full write-up with math, figures, and discussion
- `data/`  
  - `weather_albury_2016.csv` (or instructions to download from Kaggle)
- `figures/` (optional) – saved plots for double descent curves and solution fits
- `requirements.txt` – Python dependencies

---

## Methods

### 1. Synthetic Sine Dataset

- True function: \( f(x) = \sin(2 \pi x) \)
- Sampled **N = 40** points uniformly from \([-1, 1]\)
- Added uniform noise \( \epsilon \sim U(-0.1, 0.1) \)
- Fit polynomial models with degrees **0–42**
- Computed **train** and **test MSE** for each degree to visualize error vs. model complexity :contentReference[oaicite:2]{index=2}  

We then:

- Plotted **double descent curves** for OLS
- Visualized solution curves at key degrees:
  - low degree (underfitting)
  - near interpolation threshold
  - high degree (overparameterized)

### 2. Regularization

We introduced regularization to control model complexity:

- **Lasso (L1)**:  
  \[
  \min_w \|y - Xw\|_2^2 + \alpha \|w\|_1
  \]
- **Ridge (L2)**:  
  \[
  \min_w \|y - Xw\|_2^2 + \alpha \|w\|_2^2
  \]

Implementation details:

- Used `scikit-learn`’s `Lasso` and `Ridge` with fixed `alpha = 0.2`
- Repeated the degree sweep and MSE computation for both methods
- Compared:
  - shape of the double descent curve
  - stability of coefficients (via ℓ₂ norms)
  - qualitative behavior of solution curves at key polynomial degrees :contentReference[oaicite:3]{index=3}  

### 3. Real-World Weather Data

- Dataset: daily **maximum temperatures** for **Albury, Australia (2016)**, from an Australian weather Kaggle dataset
- Preprocessing:
  - Subset to one city and one year
  - Aggregated by week to create ~40 data points
  - Encoded time as “months passed” (0–12) to keep features numerically stable
- Applied the same OLS, Lasso, and Ridge pipeline using degrees **1–100**

---

## Key Findings

- **Synthetic data (sine wave)**
  - OLS exhibits a clear **double descent** curve: test error decreases, spikes near the interpolation threshold, then decreases again in the overparameterized regime.
  - **Ridge** and **Lasso** both **flatten** the spike and stabilize test error, with Ridge balancing bias/variance and Lasso encouraging sparsity but sometimes underfitting.
- **Real weather data**
  - Double descent is **less clean** but still visible in OLS through sharp test error behavior at high degrees.
  - Regularization again improves **stability and generalization**, even when the double descent curve is noisy and less textbook-like.
- Across both datasets, **regularization consistently improves robustness**, even when its exact impact on the double descent shape is more nuanced. :contentReference[oaicite:4]{index=4}  

For full equations, proofs, and plots, see the attached report: `DoubleDescentReport.pdf`.

---

## How to Run This Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/double-descent-polynomial-regression.git
   cd double-descent-polynomial-regression
