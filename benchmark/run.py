"""Run or inspect a seed-averaged checkpoint benchmark."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle

import numpy as np

from benchmark.averaging import average_across_seeds, format_seed_table
from benchmark.discovery import validate_seeds_available
from benchmark.shift import held_out_shift_configs
from configs.config import EVAL_CONFIGS, get_network_params


REPO_ROOT = Path(__file__).resolve().parents[1]
# The heavy evaluator cannot be imported by dry-run. VDN_G=0.2 is the
# explicitly selected benchmark value (its current plot default is 0.5).
DEFAULT_RHOS = {
    "VDN": [0.0, 0.008],
    "QMIX": [0.0, 0.008],
    "QTRAN": [0.0, 0.02],
    "VDN_G": [0.0, 0.2],
    "QMIX_G": [0.0, 0.2],
    "QTRAN_G": [0.0, 0.2],
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=DEFAULT_RHOS, default=["VDN", "QMIX", "VDN_G"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 10])
    parser.add_argument("--rhos", nargs="+", type=float, help="Use these rho values for every selected algorithm")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--configs", nargs="+", help="Configuration names to evaluate (default: all)")
    parser.add_argument("--held-out-shift", action="store_true", help="Use held-out shift configurations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--dry-run", action="store_true", help="Print and save the plan without evaluation")
    parser.add_argument("--smoke", action="store_true", help="Also exercise aggregation and output using synthetic rewards")
    return parser


def make_episode_seeds(eval_seed, num_episodes):
    """Match the fixed episode-seed scheme used by the existing evaluator."""
    return np.random.default_rng(eval_seed).integers(0, 200, size=num_episodes, dtype=int).tolist()


def validate_checkpoint_files(run_dir, n_agents):
    """Ensure there are enough saved weights for the requested agents."""
    net_dir = Path(run_dir) / "network_parameter"
    count = len(list(net_dir.glob("*.pt"))) if net_dir.is_dir() else 0
    if count < n_agents:
        raise ValueError(f"Missing checkpoint files in {net_dir}: need {n_agents}, found {count}")


def make_plan(args):
    """Resolve requested configs and checkpoint directories without heavy imports."""
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if len(set(args.algorithms)) != len(args.algorithms):
        raise ValueError("--algorithms must not contain duplicates")
    if args.rhos is not None and (any(not math.isfinite(rho) or rho < 0 for rho in args.rhos) or len(set(args.rhos)) != len(args.rhos)):
        raise ValueError("--rhos must be distinct, finite, and non-negative")

    configs = held_out_shift_configs() if args.held_out_shift else EVAL_CONFIGS
    if args.configs:
        by_name = {config["name"]: config for config in configs}
        unknown = [name for name in args.configs if name not in by_name]
        if unknown:
            raise ValueError(f"Unknown --configs: {', '.join(unknown)}")
        if len(set(args.configs)) != len(args.configs):
            raise ValueError("--configs must not contain duplicates")
        configs = [by_name[name] for name in args.configs]
    if not configs:
        raise ValueError("No evaluation configurations selected")

    runs = []
    for algorithm in args.algorithms:
        alg_key = algorithm.lower()
        for rho in args.rhos if args.rhos is not None else DEFAULT_RHOS[algorithm]:
            paths = validate_seeds_available(REPO_ROOT, alg_key, rho, args.seeds)
            for seed, path in paths.items():
                try:
                    validate_checkpoint_files(path, 1)
                except ValueError as exc:
                    raise ValueError(f"{algorithm} rho={rho} seed={seed}: {exc}") from exc
                runs.append({"algorithm": algorithm, "rho": rho, "seed": seed, "run_dir": str(path)})

    return {
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "configs": configs,
        "episodes": args.episodes,
        "device": args.device,
        "held_out_shift": args.held_out_shift,
        "runs": runs,
    }


def summarize_plan(plan, evaluate_run, eval_seed, smoke=False):
    """Evaluate each run and produce the existing seed-averaged pickle schema."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "eval_seed": eval_seed,
        "config_names": [config["name"] for config in plan["configs"]],
        "algorithms": {},
        "baselines": {},
        "method": "seed_averaged",
        "smoke": smoke,
    }
    groups = {}
    for run in plan["runs"]:
        key = (run["algorithm"], run["rho"])
        groups.setdefault(key, []).append(run)
    for (algorithm, rho), runs in groups.items():
        runs = sorted(runs, key=lambda run: run["seed"])
        seed_results = {run["seed"]: evaluate_run(run, plan["configs"]) for run in runs}
        means, stds, raw = average_across_seeds(seed_results)
        results["algorithms"].setdefault(algorithm, {})[rho] = {
            "rewards": means,
            "errors": stds,
            "raw_scores": raw,
            "num_seeds_evaluated": len(seed_results),
            "seed_results": seed_results,
            "seed_dirs": {run["seed"]: run["run_dir"] for run in runs},
        }
    return results


def smoke_evaluate_run(run, configs):
    """Produce deterministic synthetic rewards, never environment rewards."""
    return [run["seed"] + 10 * index + 1000 * run["rho"] for index in range(len(configs))]


def real_evaluator(plan):
    """Create the real evaluator lazily, using the repository's existing machinery."""
    import torch
    from evaluate_saved_networks import (
        ALG_NAME_TO_KEY,
        DEFAULT_SEEDS,
        _build_temp_env_for_meta,
        _load_q_nets_from_run,
        export_results_to_table,
    )
    from utils.evaluation import parallel_evaluate_policy

    for algorithm in plan["algorithms"]:
        if ALG_NAME_TO_KEY[algorithm] != algorithm.lower():
            raise ValueError(f"Checkpoint algorithm mapping changed for {algorithm}")
    device = torch.device(plan["device"])
    params = get_network_params()
    n_agents, n_actions, _ = _build_temp_env_for_meta(EVAL_CONFIGS[0])
    eval_seed = DEFAULT_SEEDS[0]
    episode_seeds = make_episode_seeds(eval_seed, plan["episodes"])

    def evaluate_run(run, configs):
        validate_checkpoint_files(run["run_dir"], n_agents)
        q_nets = _load_q_nets_from_run(run["run_dir"], n_agents, params["obs_dim"], n_actions, device)
        rewards = []
        for config in configs:
            result = parallel_evaluate_policy(
                q_nets, config, device,
                num_episodes=plan["episodes"],
                num_workers=min(4, plan["episodes"]),
                history_len=params["history_len"],
                episode_seeds=episode_seeds,
            )
            rewards.append(float(result["mean_reward"]))
        return rewards

    return evaluate_run, eval_seed, export_results_to_table


def write_results(plan, out, evaluate_run, eval_seed, smoke, export_table=None):
    results = summarize_plan(plan, evaluate_run, eval_seed, smoke)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out / f"seed_averaged_results_{stamp}.pkl"
    with results_path.open("wb") as stream:
        pickle.dump(results, stream)

    tables = []
    for algorithm, rhos in results["algorithms"].items():
        for rho, data in rhos.items():
            tables.append(format_seed_table(
                algorithm, rho, results["config_names"], data["rewards"], data["errors"],
                sorted(data["seed_results"]),
            ))
    table_path = out / "results.txt"
    if export_table is not None:
        export_table(str(results_path), output_file=table_path.name, show_raw_data=True)
        with table_path.open("a", encoding="utf-8") as stream:
            stream.write("\n\nRAW SEED-AVERAGED REWARDS\n\n" + "\n\n".join(tables) + "\n")
    else:
        table_path.write_text("SMOKE: synthetic rewards (not real evaluation results)\n\n" + "\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"Saved results: {results_path}\nSaved table: {table_path}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"Configs: {', '.join(config['name'] for config in plan['configs'])}; episodes: {plan['episodes']}")
        for run in plan["runs"]:
            print(f"{run['algorithm']} rho={run['rho']} seed={run['seed']}: {run['run_dir']}")
        print(f"Saved plan: {args.out / 'plan.json'}")
        if args.smoke:
            write_results(plan, args.out, smoke_evaluate_run, eval_seed=11, smoke=True)
        elif not args.dry_run:
            evaluate_run, eval_seed, export_table = real_evaluator(plan)
            write_results(plan, args.out, evaluate_run, eval_seed, smoke=False, export_table=export_table)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        parser.exit(1, f"benchmark: {exc}\n")


if __name__ == "__main__":
    main()
