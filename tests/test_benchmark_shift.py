from configs.config import EVAL_CONFIGS, TRAINING_CONFIG


def test_shift_configs_are_held_out_and_complete():
    from benchmark.shift import held_out_shift_configs

    configs = held_out_shift_configs()
    keys = ("building", "weather", "location")
    existing = {tuple(config[key] for key in keys) for config in EVAL_CONFIGS}
    assert configs
    assert all({"name", "building", "weather", "location", "ac_map"} <= config.keys() for config in configs)
    assert not existing.intersection(tuple(config[key] for key in keys) for config in configs)
    assert len({config["name"] for config in configs}) == len(configs)
    assert all(config["ac_map"] == TRAINING_CONFIG["ac_map"] for config in configs)
    assert all(any(config[key] != TRAINING_CONFIG[key] for config in configs) for key in keys)


def test_shift_configs_do_not_mutate_shared_training_config():
    from benchmark.shift import held_out_shift_configs

    config = held_out_shift_configs()[0]
    config["ac_map"] = "changed"
    assert TRAINING_CONFIG["ac_map"] == 1
    assert held_out_shift_configs()[0]["ac_map"] == 1
