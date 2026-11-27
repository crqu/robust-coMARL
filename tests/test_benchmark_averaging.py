import math

import pytest


def test_average_across_seeds_uses_population_std_and_sorted_seeds():
    from benchmark.averaging import average_across_seeds

    means, stds, raw = average_across_seeds({10: [10, -2], 0: [0, 4], 1: [2, 1]})
    assert means == pytest.approx([4, 1])
    assert stds == pytest.approx([math.sqrt(56 / 3), math.sqrt(6)])
    assert raw == [[0, 2, 10], [4, 1, -2]]


def test_single_seed_has_zero_std():
    from benchmark.averaging import average_across_seeds

    means, stds, raw = average_across_seeds({42: [3.5, -7]})
    assert means == [3.5, -7]
    assert stds == [0, 0]
    assert raw == [[3.5], [-7]]


@pytest.mark.parametrize("seed_results", [{}, {0: []}, {0: [1, 2], 1: [3]}])
def test_averaging_rejects_missing_or_unequal_data(seed_results):
    from benchmark.averaging import average_across_seeds

    with pytest.raises(ValueError):
        average_across_seeds(seed_results)


def test_robustness_gap_is_robust_minus_non_robust():
    from benchmark.averaging import robustness_gap

    assert robustness_gap([5, -2, 0], [3, -5, 1]) == [2, 3, -1]
    with pytest.raises(ValueError):
        robustness_gap([1, 2], [1])


def test_format_seed_table_includes_algorithm_rho_results_and_seeds():
    from benchmark.averaging import format_seed_table

    table = format_seed_table("VDN", 0.008, ["A", "B"], [1.5, -2], [0.25, 0], [0, 1, 10])
    assert "VDN" in table and "0.008" in table
    assert "A" in table and "B" in table
    assert "1.50" in table and "0.25" in table and "-2.00" in table
    assert "+-" in table and "0, 1, 10" in table
