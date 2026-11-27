"""Held-out environment configurations, with no environment instantiation."""

from configs.config import EVAL_CONFIGS, TRAINING_CONFIG


def held_out_shift_configs():
    """Return held-out shift regimes under which robust variants are measured.

    The evaluator loads one saved network per agent in the held-out env, so
    a held-out building with more agents than the six-network training
    checkpoints provide fails loudly instead of evaluating silently. Zone
    counts for the alternate buildings were not verified against a
    SustainGym install here; totals across buildings with different zone
    counts are not directly comparable.
    """
    changes = [
        {"name": "HeldOut_Warm_Humid_Atlanta", "weather": "Warm_Humid", "location": "Atlanta"},
        {"name": "HeldOut_Restaurant_Tucson", "building": "RestaurantFastFood"},
        {"name": "HeldOut_Warehouse_Albuquerque", "building": "Warehouse", "weather": "Mixed_Dry", "location": "Albuquerque"},
    ]
    keys = ("building", "weather", "location")
    used = {tuple(config[key] for key in keys) for config in EVAL_CONFIGS}
    configs = [{**TRAINING_CONFIG, **change} for change in changes]
    return [config for config in configs if tuple(config[key] for key in keys) not in used]
