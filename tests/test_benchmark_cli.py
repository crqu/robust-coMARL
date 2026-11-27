import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, cwd=ROOT):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-m", "benchmark.run", *map(str, args)],
        cwd=cwd, env=env, text=True, capture_output=True,
    )


def test_dry_run_finds_all_default_real_runs(tmp_path):
    out = tmp_path / "dry"
    result = run_cli("--dry-run", "--algorithms", "VDN", "QMIX", "VDN_G", "--seeds", 0, 1, 10, "--out", out)
    assert result.returncode == 0, result.stderr
    plan = json.loads((out / "plan.json").read_text())
    assert plan["episodes"] == 5
    assert len(plan["configs"]) == 6
    assert len(plan["runs"]) == 18
    for algorithm, robust_rho in [("VDN", 0.008), ("QMIX", 0.008), ("VDN_G", 0.2)]:
        runs = [run for run in plan["runs"] if run["algorithm"] == algorithm]
        assert {(run["seed"], run["rho"]) for run in runs} == {
            (seed, rho) for seed in (0, 1, 10) for rho in (0.0, robust_rho)
        }
        for run in runs:
            path = Path(run["run_dir"])
            assert path.is_dir() and path.parent.name == f"results_seeds_{run['seed']}"
            assert (path / "network_parameter").is_dir()
            assert str(path) in result.stdout


def test_smoke_writes_synthetic_results_and_averaged_table(tmp_path):
    out = tmp_path / "smoke"
    result = run_cli("--smoke", "--out", out)
    assert result.returncode == 0, result.stderr
    assert (out / "plan.json").is_file()
    pickles = list(out.glob("seed_averaged_results_*.pkl"))
    assert len(pickles) == 1
    with pickles[0].open("rb") as stream:
        data = pickle.load(stream)
    assert data["method"] == "seed_averaged" and data["baselines"] == {}
    assert set(data["algorithms"]) == {"VDN", "QMIX", "VDN_G"}
    assert len(data["config_names"]) == 6
    sample = data["algorithms"]["VDN"][0.008]
    assert sample["seed_results"] == {
        0: [8, 18, 28, 38, 48, 58],
        1: [9, 19, 29, 39, 49, 59],
        10: [18, 28, 38, 48, 58, 68],
    }
    assert sample["rewards"] == pytest.approx([35 / 3 + 10 * i for i in range(6)])
    assert sample["errors"] == pytest.approx([math.sqrt(182 / 9)] * 6)
    assert sample["raw_scores"][0] == [8, 9, 18]
    assert sample["num_seeds_evaluated"] == 3
    table = (out / "results.txt").read_text()
    assert "VDN" in table and "0.008" in table and "11.67" in table
    assert "0, 1, 10" in table and "SMOKE" in table.upper()


def test_raw_scores_follow_seed_results_order_for_unsorted_input(tmp_path):
    out = tmp_path / "unsorted"
    result = run_cli("--smoke", "--algorithms", "VDN", "--seeds", 10, 0, 1,
                     "--rhos", 0, "--configs", "Training_Config", "--out", out)
    assert result.returncode == 0, result.stderr
    with next(out.glob("*.pkl")).open("rb") as stream:
        sample = pickle.load(stream)["algorithms"]["VDN"][0.0]
    assert sample["raw_scores"][0] == [values[0] for values in sample["seed_results"].values()]


def test_missing_seed_exits_nonzero_with_clear_error(tmp_path):
    result = run_cli("--dry-run", "--algorithms", "VDN", "--seeds", 0, 99999, "--out", tmp_path / "missing")
    assert result.returncode != 0
    assert "missing" in result.stderr.lower() and "99999" in result.stderr and "VDN" in result.stderr.upper()


def test_custom_config_selection_and_held_out_switch(tmp_path):
    chosen = tmp_path / "chosen"
    result = run_cli("--dry-run", "--algorithms", "VDN", "--seeds", 0, "--rhos", 0, "--configs", "Training_Config", "--episodes", 3, "--out", chosen, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    plan = json.loads((chosen / "plan.json").read_text())
    assert [config["name"] for config in plan["configs"]] == ["Training_Config"]
    assert len(plan["runs"]) == 1 and plan["episodes"] == 3
    shifted = tmp_path / "shifted"
    result = run_cli("--dry-run", "--held-out-shift", "--algorithms", "VDN", "--seeds", 0, "--rhos", 0, "--out", shifted)
    assert result.returncode == 0, result.stderr
    shift_plan = json.loads((shifted / "plan.json").read_text())
    assert shift_plan["configs"] and shift_plan["configs"] != plan["configs"]


@pytest.mark.parametrize(
    ("args", "message"),
    [(('--episodes', '0'), 'episodes'), (('--rhos', '-1'), 'rho'), (('--configs', 'unknown'), 'unknown')],
)
def test_invalid_cli_options_fail_clearly(tmp_path, args, message):
    result = run_cli("--dry-run", *args, "--out", tmp_path / "bad")
    assert result.returncode != 0 and message in result.stderr.lower()


def test_episode_seeds_match_existing_evaluator_rng():
    from benchmark.run import make_episode_seeds

    assert make_episode_seeds(11, 5) == [26, 25, 159, 99, 118]
    assert make_episode_seeds(11, 5) == make_episode_seeds(11, 5)


def test_checkpoint_validation_rejects_partial_networks(tmp_path):
    from benchmark.run import validate_checkpoint_files

    net_dir = tmp_path / "network_parameter"
    net_dir.mkdir()
    (net_dir / "agent_0_q_network.pt").touch()
    with pytest.raises(ValueError, match=r"(?i)missing.*checkpoint.*2"):
        validate_checkpoint_files(tmp_path, 2)
    (net_dir / "agent_1_q_network.pt").touch()
    validate_checkpoint_files(tmp_path, 2)


@pytest.mark.parametrize("mode", ["--dry-run", "--smoke"])
def test_lightweight_modes_do_not_import_evaluation_dependencies(tmp_path, mode):
    script = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'sustaingym', 'evaluate_saved_networks'}:
        raise RuntimeError('unexpected heavyweight import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['benchmark.run', sys.argv[1], '--out', sys.argv[2]]
runpy.run_module('benchmark.run', run_name='__main__')
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, mode, str(tmp_path / "guarded")],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
