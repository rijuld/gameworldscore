"""
Configuration utilities for Oasis RL finetuning.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


CONFIG_DIR = Path(__file__).parent


def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        
    Returns:
        Dict with configuration
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override config into base config.
    
    Args:
        base: Base configuration dict
        override: Override configuration dict
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return load_config("default")

