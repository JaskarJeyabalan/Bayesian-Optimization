
# ===============================
# 1. IMPORT REQUIRED LIBRARIES
# ===============================

import numpy as np
import pandas as pd
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from scipy.stats import norm

# ===============================
# 2. LOAD REAL DATASET (CSV)
# ===============================

# Public dataset URL (NO GitHub, NO Kaggle login required)
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

# Load dataset (separator is ';')
data = pd.read_csv(DATA_URL, sep=';')

print("\nDataset Loaded Successfully")
print("Shape:", data.shape)
print(data.head())

# ===============================
# 3. SPLIT FEATURES & TARGET
# ===============================

X = data.drop("quality", axis=1)
y = data["quality"]

# ===============================
# 4. FEATURE SCALING
# ===============================

# Scaling improves Gaussian Process performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 5. OBJECTIVE FUNCTION
# ===============================

def objective_function(params):
    """
    Trains RandomForestRegressor with given hyperparameters
    and returns negative RMSE using cross-validation.
    """

    model = RandomForestRegressor(
        n_estimators=int(params[0]),
        max_depth=int(params[1]),
        min_samples_split=int(params[2]),
        max_features=params[3],
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=3,
        scoring="neg_root_mean_squared_error"
    )

    return scores.mean()

# ===============================
# 6. RANDOM HYPERPARAMETER SAMPLER
# ===============================

def random_sample():
    """
    Generates a random hyperparameter set
    """
    return np.array([
        random.randint(50, 300),     # n_estimators
        random.randint(5, 30),       # max_depth
        random.randint(2, 10),       # min_samples_split
        random.uniform(0.3, 1.0)     # max_features
    ])

# ===============================
# 7. GAUSSIAN PROCESS MODEL
# ===============================

# Matérn kernel is suitable for noisy ML problems
kernel = Matern(nu=2.5)

gp_model = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True
)

# ===============================
# 8. EXPECTED IMPROVEMENT FUNCTION
# ===============================

def expected_improvement(X_candidates, X_sample, Y_sample, gp, xi=0.01):
    """
    Computes Expected Improvement values
    """

    mu, sigma = gp.predict(X_candidates, return_std=True)
    best_value = np.max(Y_sample)

    with np.errstate(divide='warn'):
        improvement = mu - best_value - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# ===============================
# 9. BAYESIAN OPTIMIZATION LOOP
# ===============================

print("\nStarting Bayesian Optimization...\n")

# Initial random evaluations
X_sample = np.array([random_sample() for _ in range(5)])
Y_sample = np.array([objective_function(x) for x in X_sample])

bo_scores = []
BO_ITERATIONS = 50

for i in range(BO_ITERATIONS):

    # Fit GP model
    gp_model.fit(X_sample, Y_sample)

    # Generate candidate points
    X_candidates = np.array([random_sample() for _ in range(300)])

    # Calculate Expected Improvement
    ei = expected_improvement(X_candidates, X_sample, Y_sample, gp_model)

    # Select best candidate
    next_x = X_candidates[np.argmax(ei)]
    next_y = objective_function(next_x)

    # Update observations
    X_sample = np.vstack((X_sample, next_x))
    Y_sample = np.append(Y_sample, next_y)

    # Track best score
    best_score = np.max(Y_sample)
    bo_scores.append(best_score)

    print(
        f"BO Iteration {i+1:02d} | "
        f"Best RMSE so far: {-best_score:.4f}"
    )

# ===============================
# 10. RANDOM SEARCH BASELINE
# ===============================

print("\nStarting Random Search...\n")

random_scores = []
RS_ITERATIONS = 50

for i in range(RS_ITERATIONS):
    params = random_sample()
    score = objective_function(params)

    if i == 0:
        random_scores.append(score)
    else:
        random_scores.append(max(random_scores[-1], score))

    print(
        f"RS Iteration {i+1:02d} | "
        f"Best RMSE so far: {-random_scores[-1]:.4f}"
    )

# ===============================
# 11. CONVERGENCE ANALYSIS
# ===============================

best_bo = max(bo_scores)
best_rs = max(random_scores)

print("\n===== FINAL RESULTS =====")
print("Best Bayesian Optimization RMSE:", -best_bo)
print("Best Random Search RMSE:", -best_rs)

# ===============================
# 12. ITERATIONS TO REACH 90%
# ===============================

bo_threshold = 0.9 * best_bo
rs_threshold = 0.9 * best_rs

bo_90 = next(i for i, v in enumerate(bo_scores) if v >= bo_threshold) + 1
rs_90 = next(i for i, v in enumerate(random_scores) if v >= rs_threshold) + 1

print("\nIterations to reach 90% of best performance:")
print("Bayesian Optimization:", bo_90)
print("Random Search:", rs_90)

# ===============================
# 13. BEST HYPERPARAMETERS
# ===============================

best_index = np.argmax(Y_sample)
best_params = X_sample[best_index]

print("\nBest Hyperparameters Found by Bayesian Optimization:")
print("n_estimators:", int(best_params[0]))
print("max_depth:", int(best_params[1]))
print("min_samples_split:", int(best_params[2]))
print("max_features:", best_params[3])

print("\nPROJECT COMPLETED SUCCESSFULLY")
