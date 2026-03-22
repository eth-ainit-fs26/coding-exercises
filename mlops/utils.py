"""
Utility functions for MLOps drift simulation.

This module contains reusable functions for:
- Data generation with drift
- Drift scheduling
- Visualization of simulation results
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_batch(mu_shift=0.0, size=400):
    """
    Generate a batch of binary classification data with drift.

    Returns a batch (X, y) where class 0 is centered near (0,0)
    and class 1 near (2+mu_shift, 2+mu_shift).

    Parameters:
    -----------
    mu_shift : float
        The drift amount to shift class 1's center
    size : int
        Total number of samples (split equally between classes)

    Returns:
    --------
    X : np.ndarray
        Feature matrix of shape (size, 2)
    y : np.ndarray
        Labels of shape (size,)
    """
    half = size // 2
    # Class 0 (stationary-ish)
    X0 = np.random.normal(loc=[0.0, 0.0], scale=[1.0, 1.0], size=(half, 2))
    y0 = np.zeros(half, dtype=int)
    # Class 1 (drifting)
    X1 = np.random.normal(loc=[2.0 + mu_shift, 2.0 + mu_shift], scale=[1.0, 1.0], size=(half, 2))
    y1 = np.ones(half, dtype=int)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    return X, y


def drift_schedule(round_idx, total_rounds, step_scale=0.05):
    """
    Calculate drift step using random walk that tends to increase over time.

    Parameters:
    -----------
    round_idx : int
        Current round index
    total_rounds : int
        Total number of rounds in simulation
    step_scale : float
        Scale parameter for the random step variance

    Returns:
    --------
    float
        The drift step to add to current mu
    """
    step = np.random.normal(loc=0.03, scale=step_scale * (1 + round_idx / total_rounds))
    return step


def plot_drift_over_time(drift_values, rounds):
    """
    Plot data drift (mu) over rounds.

    Parameters:
    -----------
    drift_values : list
        List of drift values for each round
    rounds : int
        Total number of rounds
    """
    plt.figure()
    plt.plot(range(1, rounds + 1), drift_values)
    plt.title("Data Drift (mu) over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Shift (mu)")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_model_accuracies(acc_A_hist, acc_B_hist, retrain_rounds, promotions, rounds):
    """
    Plot model accuracies over time with retrain and promotion markers.

    Parameters:
    -----------
    acc_A_hist : list
        Accuracy history for Model A
    acc_B_hist : list
        Accuracy history for Model B
    retrain_rounds : list
        Rounds where Model B was retrained
    promotions : list
        Rounds where champion promotion occurred
    rounds : int
        Total number of rounds
    """
    plt.figure()
    plt.plot(range(1, rounds + 1), acc_A_hist, label="Model A (static)", linewidth=2)
    plt.plot(range(1, rounds + 1), acc_B_hist, label="Model B (retrained)", linewidth=2)

    # Mark retraining rounds
    for rr in retrain_rounds:
        plt.axvline(rr, linestyle=':', alpha=0.6, color='gray')

    # Mark promotion rounds
    for pr in promotions:
        plt.axvline(pr, linestyle='--', alpha=0.8, color='red', linewidth=1.5)

    plt.title("Accuracy over Rounds (vertical: retrain ':', promotion '--')")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_served_accuracy(served_acc_hist, rounds):
    """
    Plot blended (served) accuracy over rounds.

    This shows the actual accuracy experienced by users
    when using canary routing.

    Parameters:
    -----------
    served_acc_hist : list
        History of blended accuracy values
    rounds : int
        Total number of rounds
    """
    plt.figure()
    plt.plot(range(1, rounds + 1), served_acc_hist, linewidth=2, color='green')
    plt.title("Served (Blended) Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Served Accuracy")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_sample_data(X_sample, y_sample):
    """
    Visualize a sample of the binary classification data.

    Shows the two classes in a 2D scatter plot to help understand
    the data distribution before drift occurs.

    Parameters:
    -----------
    X_sample : np.ndarray
        Feature matrix of shape (n_samples, 2)
    y_sample : np.ndarray
        Labels of shape (n_samples,)
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X_sample[y_sample == 0, 0], X_sample[y_sample == 0, 1],
                alpha=0.5, label='Class 0 (stationary)', s=30)
    plt.scatter(X_sample[y_sample == 1, 0], X_sample[y_sample == 1, 1],
                alpha=0.5, label='Class 1 (will drift)', s=30)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Sample Data (No Drift)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
