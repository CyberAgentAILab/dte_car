"""
Simulation script for Distributional Treatment Effects (DTE) estimation under covariate-adaptive randomization.

This script implements simulations for comparing different DTE estimators:
- Simple stratified estimator
- Linear adjusted estimator
- ML (XGBoost) adjusted estimator
"""

import argparse
import random
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import dte_adj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from sklearn.linear_model import LinearRegression

# Configuration constants
RANDOM_SEED = 123
TOTAL_ITERATIONS = 1000
DIMENSION = 20
DEFAULT_STRATA = 4
TREATMENT_ARM = 1
CONTROL_ARM = 0

# Plot styling
COLORS = {"Empirical": "green", "Linear": "purple", "XGBoost": "orange"}
FONT_SIZE = 18


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulation script for DTE adjustment methods comparison."
    )
    parser.add_argument(
        "--n", type=int, default=1000,
        help="Sample size for data generation (default: 1000)"
    )
    parser.add_argument(
        "--discrete", type=str, default="false",
        help="Generate discrete outcomes (true/false, default: false)"
    )
    parser.add_argument(
        "--discrete_covariates", type=str, default="false",
        help="Generate discrete covariates (true/false, default: false)"
    )
    parser.add_argument(
        "--iterations", type=int, default=TOTAL_ITERATIONS,
        help=f"Number of simulation iterations (default: {TOTAL_ITERATIONS})"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    return parser.parse_args()


def generate_data(
    n: int = 1000,
    num_strata: int = DEFAULT_STRATA,
    dimension: int = DIMENSION,
    discrete: bool = False,
    discrete_covariates: bool = False
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data following the DGP specification.

    Args:
        n: Sample size
        num_strata: Number of strata for randomization
        dimension: Dimension of covariate vector
        discrete: Whether to generate discrete outcomes
        discrete_covariates: Whether to generate discrete covariates

    Returns:
        Dictionary containing generated data arrays
    """
    # Generate stratification variable Z_i ~ U(0,1)
    Z = np.random.uniform(0, 1, n)

    # Define strata S_i based on Z quantiles
    strata_boundaries = np.linspace(0, 1, num_strata + 1)[1:-1]
    S_i = np.digitize(Z, strata_boundaries)

    # Generate covariates X_i
    if discrete_covariates:
        X = np.random.uniform(-5, 5, size=(n, dimension)).round()
    else:
        X = np.random.multivariate_normal(
            mean=np.zeros(dimension),
            cov=np.eye(dimension),
            size=n
        )

    # Treatment assignment within strata (balanced randomization)
    W = _assign_treatment_within_strata(S_i, n)

    # Generate outcomes using the specified DGP
    Y = _generate_outcomes(X, W, Z, discrete)

    return {
        'W': W, 'X': X, 'Z': Z, 'Y': Y, 'strata': S_i
    }


def _assign_treatment_within_strata(strata: np.ndarray, n: int) -> np.ndarray:
    """Assign treatment within each stratum with exact balance."""
    W = np.zeros(n, dtype=int)

    for s in np.unique(strata):
        stratum_indices = np.where(strata == s)[0]
        n_stratum = len(stratum_indices)

        # Assign exactly half to treatment
        W[stratum_indices[:n_stratum // 2]] = 1

        # Shuffle within stratum to randomize assignment
        np.random.shuffle(W[stratum_indices])

    return W


def _generate_outcomes(
    X: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    discrete: bool
) -> np.ndarray:
    """Generate outcomes according to the DGP specification."""
    # Baseline outcome function
    b_X = (np.sin(np.pi * X[:, 0] * X[:, 1]) +
           2 * (X[:, 2] - 0.5)**2 +
           X[:, 3] +
           0.5 * X[:, 4])

    # Treatment effect function
    c_X = 0.1 * (X[:, 0] + np.log(1 + np.exp(X[:, 1])))

    # Parameters and noise
    gamma = 0.1
    noise = np.random.normal(0, 1, len(X))

    # Generate continuous outcomes
    Y = b_X + c_X * W + gamma * Z + noise

    # Convert to discrete if requested
    if discrete:
        Y = np.random.poisson(0.2 * np.abs(Y))

    return Y


def create_xgb_regressor() -> xgb.XGBRegressor:
    """Create XGBoost regressor with optimized hyperparameters."""
    return xgb.XGBRegressor(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED
    )


def run_single_simulation(
    n: int,
    discrete: bool,
    discrete_covariates: bool,
    locations: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Run a single simulation iteration."""
    data = generate_data(n=n, discrete=discrete, discrete_covariates=discrete_covariates)
    X, Y, W, S = data["X"], data["Y"], data["W"], data["strata"]

    # Augment features with treatment indicator
    X_augmented = np.hstack([X, W.reshape(-1, 1)])

    results = {}
    execution_times = {}

    # Empirical estimator
    start_time = time.time()
    empirical_estimator = dte_adj.SimpleStratifiedDistributionEstimator()
    empirical_estimator.fit(X_augmented, W, Y, S)
    results['empirical'] = empirical_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['empirical'] = time.time() - start_time

    # Linear adjusted estimator
    start_time = time.time()
    linear_estimator = dte_adj.AdjustedStratifiedDistributionEstimator(
        LinearRegression(), is_multi_task=False, folds=2
    )
    linear_estimator.fit(X_augmented, W, Y, S)
    results['linear'] = linear_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['linear'] = time.time() - start_time

    # XGBoost adjusted estimator
    start_time = time.time()
    xgb_estimator = dte_adj.AdjustedStratifiedDistributionEstimator(
        create_xgb_regressor(), is_multi_task=False, folds=2
    )
    xgb_estimator.fit(X_augmented, W, Y, S)
    results['xgb'] = xgb_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )
    execution_times['xgb'] = time.time() - start_time

    return results, execution_times


def calculate_metrics(
    results: Dict[str, List],
    true_dte: np.ndarray,
    locations: np.ndarray
) -> pd.DataFrame:
    """Calculate performance metrics from simulation results."""
    metrics_data = {"locations": locations}

    for method in ['empirical', 'linear', 'xgb']:
        method_results = np.array(results[method])
        point_estimates = method_results[:, 0]
        lower_bounds = method_results[:, 1]
        upper_bounds = method_results[:, 2]

        # Calculate metrics
        interval_lengths = (upper_bounds - lower_bounds).mean(axis=0)
        coverage_prob = ((upper_bounds >= true_dte) & (true_dte >= lower_bounds)).mean(axis=0)
        rmse = np.sqrt(((point_estimates - true_dte) ** 2).mean(axis=0))

        metrics_data.update({
            f"interval length - {method}": interval_lengths,
            f"coverage probability - {method}": coverage_prob,
            f"RMSE - {method}": rmse
        })

    df = pd.DataFrame(metrics_data)

    # Calculate RMSE reductions
    df["RMSE reduction (%) linear / empirical"] = (
        1 - df["RMSE - linear"] / df["RMSE - empirical"]
    ) * 100
    df["RMSE reduction (%) xgb / empirical"] = (
        1 - df["RMSE - xgb"] / df["RMSE - empirical"]
    ) * 100

    return df


def create_performance_plots(df: pd.DataFrame, locations: np.ndarray, n: int) -> None:
    """Create visualization plots for simulation results."""
    # Performance metrics comparison
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    metrics = {
        'RMSE': {
            "Empirical": df["RMSE - empirical"],
            "Linear": df["RMSE - linear"],
            "XGBoost": df["RMSE - xgb"]
        },
        'Average CI Length': {
            "Empirical": df["interval length - empirical"],
            "Linear": df["interval length - linear"],
            "XGBoost": df["interval length - xgb"]
        },
        'Coverage Probability': {
            "Empirical": df["coverage probability - empirical"],
            "Linear": df["coverage probability - linear"],
            "XGBoost": df["coverage probability - xgb"]
        }
    }

    for i, (title, data) in enumerate(metrics.items()):
        ax = axs[i]
        for label, values in data.items():
            ax.plot(locations, values, label=label, marker='o', color=COLORS[label])
        ax.set_title(title, fontsize=FONT_SIZE)
        ax.set_xlabel("Y", fontsize=FONT_SIZE)
        if i == 0:
            ax.set_ylabel("Value", fontsize=FONT_SIZE)
        ax.grid(True)

    fig.legend(
        ['Empirical', 'Linear adjustment', 'ML adjustment'],
        loc='lower center', ncol=3, fontsize=FONT_SIZE
    )
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.show()

    # RMSE reduction plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        locations, df["RMSE reduction (%) linear / empirical"],
        color='purple', marker='o', label='Linear adjustment'
    )
    plt.plot(
        locations, df["RMSE reduction (%) xgb / empirical"],
        color='orange', marker='o', label='ML adjustment'
    )
    plt.axhline(y=0, color='black', linewidth=1)
    plt.title(f"RMSE Reduction: Adjusted vs Empirical (n={n})", fontsize=FONT_SIZE)
    plt.xlabel("Y", fontsize=FONT_SIZE)
    plt.ylabel("RMSE Reduction (%)", fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main simulation execution function."""
    args = parse_arguments()

    # Set parameters
    n = args.n
    is_discrete = args.discrete.lower() == "true"
    is_discrete_covariates = args.discrete_covariates.lower() == "true"
    iterations = args.iterations

    # Create output label
    discrete_label = "discrete" if is_discrete else "continuous"
    if is_discrete_covariates:
        discrete_label += "_covariates"

    print(f"Starting simulation with n={n}, discrete={is_discrete}, "
          f"discrete_covariates={is_discrete_covariates}, iterations={iterations}")

    # Set random seed
    set_random_seed(args.seed)

    # Generate large test dataset for ground truth
    print("Generating ground truth DTE...")
    test_data = generate_data(
        n=10**6, discrete=is_discrete, discrete_covariates=is_discrete_covariates
    )
    X_test, Y_test, W_test, S_test = (
        test_data["X"], test_data["Y"], test_data["W"], test_data["strata"]
    )

    # Define evaluation locations
    if is_discrete:
        locations = np.arange(1, 7)
    else:
        locations = np.array([np.quantile(Y_test, i * 0.1) for i in range(1, 10)])

    # Calculate ground truth DTE
    ground_truth_estimator = dte_adj.SimpleStratifiedDistributionEstimator()
    ground_truth_estimator.fit(X_test, W_test, Y_test, S_test)
    true_dte, _, _ = ground_truth_estimator.predict_dte(
        TREATMENT_ARM, CONTROL_ARM, locations, variance_type="moment"
    )

    # Run simulation iterations
    print(f"Running {iterations} simulation iterations...")
    results = defaultdict(list)
    execution_times = defaultdict(list)

    for epoch in tqdm.tqdm(range(iterations)):
        iter_results, iter_times = run_single_simulation(
            n, is_discrete, is_discrete_covariates, locations
        )

        for method in ['empirical', 'linear', 'xgb']:
            results[method].append(iter_results[method])
            execution_times[method].append(iter_times[method])

    # Calculate and save metrics
    print("Calculating metrics...")
    df = calculate_metrics(results, true_dte, locations)
    output_file = f"dte_{n}_{discrete_label}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print summary statistics
    print("\nExecution time summary (seconds):")
    for method in ['empirical', 'linear', 'xgb']:
        times = execution_times[method]
        print(f"{method:>10}: mean={np.mean(times):.4f}, std={np.std(times):.4f}")

    print("\nAverage RMSE reductions:")
    print(f"Linear vs Empirical: {df['RMSE reduction (%) linear / empirical'].mean():.2f}%")
    print(f"XGBoost vs Empirical: {df['RMSE reduction (%) xgb / empirical'].mean():.2f}%")

    # Create visualizations
    print("Creating plots...")
    create_performance_plots(df, locations, n)

    print("Simulation completed successfully!")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress sklearn/xgboost warnings
        main()
