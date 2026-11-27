"""Discover checkpoint directories without importing the evaluation stack."""

import os
import re


# Kept in sync with the evaluator, which cannot be imported without torch.
RUN_DIR_REGEX = re.compile(r"^robust_(?P<alg>[a-z0-9_]+)_rho_(?P<rho>[0-9.]+)_(?P<ts>\d{8}_\d{6})$")
SEED_DIR_REGEX = re.compile(r"^results_seeds_(?P<seed>\d+)$")


def parse_run_dir(name):
    """Parse a checkpoint directory name, returning None for invalid names."""
    match = RUN_DIR_REGEX.match(name)
    if not match:
        return None
    try:
        rho = float(match.group("rho"))
    except ValueError:
        return None
    return {"alg": match.group("alg"), "rho": rho, "timestamp": match.group("ts")}


def latest_run_per_rho(results_root, alg_key):
    """Return the latest timestamped directory for each rho of an algorithm."""
    if not os.path.isdir(results_root):
        return {}
    latest = {}
    timestamps = {}
    for name in sorted(os.listdir(results_root)):
        run = parse_run_dir(name)
        path = os.path.join(results_root, name)
        if not run or run["alg"] != alg_key or not os.path.isdir(path):
            continue
        rho = run["rho"]
        if rho not in timestamps or run["timestamp"] > timestamps[rho]:
            latest[rho] = path
            timestamps[rho] = run["timestamp"]
    return latest


def discover_seed_runs(repo_root, alg_key, rho):
    """Return the latest matching checkpoint directory per training seed."""
    if not os.path.isdir(repo_root):
        return {}
    found = {}
    for name in sorted(os.listdir(repo_root)):
        match = SEED_DIR_REGEX.match(name)
        seed_root = os.path.join(repo_root, name)
        if not match or not os.path.isdir(seed_root):
            continue
        candidates = [
            path for value, path in latest_run_per_rho(seed_root, alg_key).items()
            if abs(value - rho) < 1e-6
        ]
        if candidates:
            found[int(match.group("seed"))] = max(
                candidates, key=lambda path: parse_run_dir(os.path.basename(path))["timestamp"]
            )
    return found


def validate_seeds_available(repo_root, alg_key, rho, seeds):
    """Return requested runs or identify all missing checkpoint seeds."""
    found = discover_seed_runs(repo_root, alg_key, rho)
    missing = sorted(set(seeds) - found.keys())
    if missing:
        raise ValueError(f"Missing checkpoint seeds for {alg_key} rho={rho}: {', '.join(map(str, missing))}")
    return {seed: found[seed] for seed in seeds}
