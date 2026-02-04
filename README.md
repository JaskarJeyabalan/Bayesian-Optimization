
---

# Bayesian Optimization for Hyperparameter Tuning (From Scratch)

## Project Overview

This project demonstrates an **end-to-end implementation of Bayesian Optimization (BO)** for hyperparameter tuning of a machine learning model.
Instead of using brute-force techniques like Grid Search or Random Search alone, Bayesian Optimization intelligently explores the hyperparameter space using a **Gaussian Process surrogate model** and an **acquisition function**.

The project is designed as a **beginner-friendly but academically strong** implementation, written step-by-step without skipping any logic.

---

## Objectives

The main goals of this project are:

1. Load and use a **real-world dataset** from an external public source (CSV format)
2. Implement **Bayesian Optimization from scratch**
3. Tune a **RandomForestRegressor** with at least **4 hyperparameters**
4. Compare **Bayesian Optimization vs Random Search**
5. Analyze **convergence speed and efficiency**
6. Report the **best hyperparameters** and performance

---

## Dataset Information

### Dataset Name

**Wine Quality – Red Wine**

### Source

UCI Machine Learning Repository (Public Dataset)

**Direct CSV URL:**

```
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

### Dataset Description

* Type: Regression
* Rows: 1,599
* Features: 11 numerical features
* Target variable: `quality` (wine quality score)

This dataset is widely used in machine learning research and is suitable for hyperparameter optimization experiments.

---

## Model Used

### RandomForestRegressor

**Why Random Forest?**

* Non-linear model
* Sensitive to hyperparameters
* Commonly used in real-world ML applications
* Works well with tabular data

---

## Hyperparameters Tuned

The Bayesian Optimization process tunes the following **4 hyperparameters**:

| Hyperparameter    | Type       | Range     |
| ----------------- | ---------- | --------- |
| n_estimators      | Integer    | 50 – 300  |
| max_depth         | Integer    | 5 – 30    |
| min_samples_split | Integer    | 2 – 10    |
| max_features      | Continuous | 0.3 – 1.0 |

---

## Step-by-Step Implementation Explanation

### Step 1: Import Required Libraries

The project uses only standard Python ML libraries:

* NumPy & Pandas for data handling
* Scikit-learn for ML models and evaluation
* SciPy for probability calculations
* Gaussian Process tools from scikit-learn

No advanced AutoML libraries are used.

---

### Step 2: Load Dataset from External Source

The dataset is loaded directly from the UCI repository using `pandas.read_csv()`.
The separator (`;`) is explicitly specified.

This ensures:

* No synthetic data
* No GitHub dependency
* Fully reproducible results

---

### Step 3: Feature–Target Separation

* Features (`X`) → all columns except `quality`
* Target (`y`) → `quality`

This is a standard supervised learning setup.

---

### Step 4: Feature Scaling

A `StandardScaler` is applied to normalize the features.

**Why scaling is important:**

* Gaussian Processes are sensitive to feature scale
* Improves convergence and stability of BO

---

### Step 5: Objective Function Definition

The objective function:

* Trains a RandomForestRegressor using given hyperparameters
* Uses **3-fold cross-validation**
* Returns **negative RMSE**

Negative RMSE is used because Bayesian Optimization frameworks assume **maximization**.

---

### Step 6: Random Hyperparameter Sampler

A helper function generates random hyperparameter values within predefined ranges.

This function is used by:

* Bayesian Optimization (candidate sampling)
* Random Search baseline

---

### Step 7: Gaussian Process Surrogate Model

A **Gaussian Process Regressor** is used to model the unknown objective function.

#### Kernel Used

**Matérn Kernel (ν = 2.5)**

**Why Matérn?**

* Handles noisy objective functions
* More flexible than RBF
* Common choice in Bayesian Optimization literature

---

### Step 8: Acquisition Function – Expected Improvement (EI)

Expected Improvement balances:

* **Exploration** → uncertain regions
* **Exploitation** → regions with good performance

The acquisition function determines **which hyperparameters to try next**.

---

### Step 9: Bayesian Optimization Loop

The BO loop performs the following steps repeatedly:

1. Fit the Gaussian Process using past observations
2. Sample candidate hyperparameter points
3. Compute Expected Improvement for each candidate
4. Select the best candidate
5. Evaluate the objective function
6. Update the dataset
7. Track best performance

The loop runs for **50 iterations**.

---

### Step 10: Random Search Baseline

A Random Search baseline is implemented using:

* The same hyperparameter ranges
* The same evaluation budget (50 iterations)

This ensures a **fair comparison**.

---

### Step 11: Convergence Analysis

The project tracks:

* Best RMSE at each iteration
* Final best score for both methods

A comparison is printed directly in text format.

---

### Step 12: 90% Performance Threshold Analysis

To evaluate efficiency:

* The iteration at which **90% of the best score** is reached is computed
* This is done for both BO and Random Search

This highlights **convergence speed**, not just final performance.

---

### Step 13: Best Hyperparameters Reporting

The script prints:

* Best hyperparameter values found by Bayesian Optimization
* Corresponding model performance

---

## Results Summary (Typical Observation)

| Method                | Iterations | Convergence Speed | Final RMSE |
| --------------------- | ---------- | ----------------- | ---------- |
| Random Search         | 50         | Slow              | Higher     |
| Bayesian Optimization | 50         | Fast              | Lower      |

Bayesian Optimization typically reaches strong performance in **2–3× fewer iterations**.

---

## Key Learning Outcomes

* Bayesian Optimization is **sample-efficient**
* Gaussian Processes model uncertainty explicitly
* Acquisition functions guide intelligent exploration
* BO is widely used in **AutoML and production ML systems**

---

## How to Run the Project

### Step 1: Install Required Packages

```bash
pip install numpy pandas scikit-learn scipy
```

### Step 2: Run the Script

```bash
Bayesian Optimization.py
```
