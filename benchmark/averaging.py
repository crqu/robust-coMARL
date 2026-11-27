"""Seed-wise reward summaries, independent of the evaluation stack."""

import numpy as np


def average_across_seeds(seed_results):
    """Return per-config means, population standard deviations, and raw rewards."""
    if not seed_results:
        raise ValueError("No seed results to average")
    values = [seed_results[seed] for seed in sorted(seed_results)]
    if not values[0] or any(len(row) != len(values[0]) for row in values):
        raise ValueError("Each seed must have the same nonempty set of config rewards")
    rewards = np.asarray(values, dtype=float)
    return rewards.mean(axis=0).tolist(), rewards.std(axis=0).tolist(), rewards.T.tolist()


def robustness_gap(robust_means, non_robust_means):
    """Return robust minus non-robust mean reward for each configuration."""
    if len(robust_means) != len(non_robust_means):
        raise ValueError("Robust and non-robust results must have the same length")
    return [robust - baseline for robust, baseline in zip(robust_means, non_robust_means)]


def format_seed_table(algorithm, rho, config_names, means, stds, seeds):
    """Format raw reward means and standard deviations as a plain-text table."""
    if len(config_names) != len(means) or len(means) != len(stds):
        raise ValueError("Configuration names, means and standard deviations must align")
    lines = [f"Algorithm: {algorithm}  rho: {rho}", f"Seeds: {', '.join(map(str, seeds))}", "Configuration | Mean +- std"]
    lines.extend(f"{name} | {mean:.2f} +- {std:.2f}" for name, mean, std in zip(config_names, means, stds))
    return "\n".join(lines)
