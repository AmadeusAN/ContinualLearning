import argparse
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """从 YAML 文件中读取配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config: dict, overrides: dict = None) -> argparse.Namespace:
    """
    将 dict 配置转为 argparse.Namespace，
    并允许通过 overrides 字典覆盖其中的值。
    """
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v
    return argparse.Namespace(**config)
