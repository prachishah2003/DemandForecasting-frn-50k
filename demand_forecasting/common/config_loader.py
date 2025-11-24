# demand_forecasting/common/config_loader.py
# =============================================================

import yaml
from types import SimpleNamespace


def dict_to_namespace(d: dict):
    """
    Recursively convert dictionaries into dot-accessible namespaces.
    Example:
        cfg.models.chronos.fine_tune.steps
    """
    if not isinstance(d, dict):
        return d

    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str):
    """
    Load a YAML config file and convert it into a nested namespace object.
    """
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return dict_to_namespace(cfg_dict)
