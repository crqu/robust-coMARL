from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("robust_vdn_rho_0.008_20250917_221551", {"alg": "vdn", "rho": 0.008, "timestamp": "20250917_221551"}),
        ("robust_vdn_g_rho_0.2_20250917_235048", {"alg": "vdn_g", "rho": 0.2, "timestamp": "20250917_235048"}),
        ("robust_qmix_groupdr_rho_0.0_20250101_000000", {"alg": "qmix_groupdr", "rho": 0.0, "timestamp": "20250101_000000"}),
        ("robust_vdn_rho_0..2_20250917_235048", None),
        ("robust_vdn_rho_-0.2_20250917_235048", None),
        ("robust_VDN_rho_0.2_20250917_235048", None),
        ("robust_vdn_rho_0.2_20250917", None),
        ("robust_vdn_rho_0.2_20250917_235048_extra", None),
    ],
)
def test_parse_run_dir(name, expected):
    from benchmark.discovery import parse_run_dir

    assert parse_run_dir(name) == expected


def test_latest_run_per_rho_ignores_other_algorithms_and_files(tmp_path):
    from benchmark.discovery import latest_run_per_rho

    for name in (
        "robust_vdn_rho_0.0_20250101_100000",
        "robust_vdn_rho_0.0_20250102_090000",
        "robust_vdn_rho_0.008_20250101_100000",
        "robust_vdn_g_rho_0.0_20250103_100000",
    ):
        (tmp_path / name).mkdir()
    (tmp_path / "robust_vdn_rho_0.0_20260101_000000").touch()
    found = latest_run_per_rho(tmp_path, "vdn")
    assert {rho: Path(path).name for rho, path in found.items()} == {
        0.0: "robust_vdn_rho_0.0_20250102_090000",
        0.008: "robust_vdn_rho_0.008_20250101_100000",
    }
    assert latest_run_per_rho(tmp_path / "missing", "vdn") == {}


def test_discover_seed_runs_selects_latest_and_matches_rho(tmp_path):
    from benchmark.discovery import discover_seed_runs

    for seed, name in [
        (0, "robust_vdn_rho_0.008_20250101_100000"),
        (0, "robust_vdn_rho_0.008_20250102_100000"),
        (1, "robust_vdn_rho_0.008_20250101_100000"),
        (10, "robust_vdn_rho_0.0_20250101_100000"),
    ]:
        (tmp_path / f"results_seeds_{seed}" / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "results_seeds_bad" / "robust_vdn_rho_0.008_20250101_100000").mkdir(parents=True)
    found = discover_seed_runs(tmp_path, "vdn", 0.008)
    assert set(found) == {0, 1}
    assert Path(found[0]).name == "robust_vdn_rho_0.008_20250102_100000"
    assert Path(found[1]).parent.name == "results_seeds_1"
    assert discover_seed_runs(tmp_path / "missing", "vdn", 0.008) == {}


def test_missing_seed_error_lists_only_requested_missing_seeds(tmp_path):
    from benchmark.discovery import validate_seeds_available

    (tmp_path / "results_seeds_1" / "robust_vdn_rho_0.0_20250101_100000").mkdir(parents=True)
    with pytest.raises(ValueError, match=r"(?i)missing.*0.*10") as error:
        validate_seeds_available(tmp_path, "vdn", 0.0, [0, 1, 10])
    assert "vdn" in str(error.value) and "0.0" in str(error.value)
    found = validate_seeds_available(tmp_path, "vdn", 0.0, [1])
    assert list(found) == [1]
