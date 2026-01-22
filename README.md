# Double Descent in Polynomial Regression

This project explores the **double descent** phenomenon using synthetic sinusoidal data and real-world weather data, with **Ridge** and **Lasso** regularization.

📓 View the notebook: [DoubleDescent.ipynb](DoubleDescent.ipynb)  
📄 Final report: [DoubleDescentReport.pdf](DoubleDescentReport.pdf)

---

## Project Overview

Classical learning theory predicts a **U-shaped bias–variance tradeoff**: test error decreases with model complexity, then increases as the model overfits. The double descent phenomenon describes a more modern behavior where test error:

1. **Decreases** as model complexity increases (underparameterized regime)  
2. **Spikes** near the interpolation threshold (model just complex enough to fit the training data)  
3. **Decreases again** in the overparameterized regime  

In this project, we:

- Recreate double descent using **polynomial regression** on a noisy sine wave
- Introduce **L1 (Lasso)** and **L2 (Ridge)** regularization
- Apply the same pipeline to **real daily maximum temperatures** from Albury, Australia
- Compare behavior across model complexities using **MSE** curves and **solution plots**

---

## Repository Contents

- `DoubleDescent.ipynb` — main analysis notebook (synthetic + weather experiments)
- `DoubleDescentReport.pdf` — full write-up with math, figures, and discussion
- `data/`
  - `weather_albury_2016.csv` (Kaggle dataset)

---

## Methods

### 1. Synthetic Sine Dataset

We generated a synthetic dataset from the function:

$$
f(x) = \sin(2\pi x)
$$

with the following setup:

- Sampled **N = 40** points uniformly from $[-1, 1]$
- Added uniform noise $\epsilon \sim U(-0.1, 0.1)$
- Fit polynomial models with degrees **0–42**
- Computed **train** and **test** MSE for each degree to visualize performance vs. model complexity

We then:

- Plotted **double descent** curves for ordinary least squares (OLS)
- Visualized solution curves at key degrees:
  - Low degree (underfitting)
  - Near interpolation threshold
  - High degree (overparameterized)

---

### 2. Regularization

To control model complexity, we introduced **L1** and **L2** regularization.

- **Lasso (L1)** objective:

  $
  \min_w |y - Xw|_2^2 + \alpha |w|_1
  $

- **Ridge (L2)** objective:

  $
  \min_w |y - Xw|_2^2 + \alpha |w|_2^2
  $

Implementation details:

- Used `scikit-learn`'s `Lasso` and `Ridge` with fixed $\alpha = 0.2$
- Repeated polynomial degree sweep for both methods
- Compared:
  - **Shape** of the double descent curve
  - **Stability** of coefficients (via $\ell_2$ norms)
  - **Solution curves** at key polynomial degrees

---

### 3. Real-World Weather Data

We repeated the experiment on real data:

- Dataset: daily **maximum temperatures** for **Albury, Australia (2016)** (Kaggle)
- Preprocessing steps:
  - Filtered to one city and one year
  - Aggregated values by week to obtain ~40 points
  - Encoded time as “months passed” (0–12) for numerical stability
- Applied **OLS**, **Lasso**, and **Ridge** using polynomial degrees **1–100**

---

## Key Findings

- **Synthetic data (sine wave):**
  - OLS shows a clear **double descent** curve: decreasing → spiking → decreasing again.
  - **Ridge** and **Lasso** both **flatten** the spike and improve stability, with Ridge balancing bias–variance and Lasso inducing sparsity.

- **Real weather data:**
  - Double descent is **less clean** but still visible through sharp test MSE behavior at high degrees.
  - Regularization improves **stability** and **generalization**, even when the double descent shape is noisy.

**Overall:** Across both datasets, **regularization improves robustness**, even though its exact effect on double descent varies with data distribution.

For full equations, proofs, and figures, refer to the report: `DoubleDescentReport.pdf`.

---

## How to Run This Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/double-descent-polynomial-regression.git
   cd double-descent-polynomial-regression
