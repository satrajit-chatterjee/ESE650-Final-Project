from typing import List, Optional
from dataclasses import dataclass, field
from f1tenth_gym.envs.track import Track

import numpy as np

import yaml

@dataclass
class EnvConfig():
    l_path: float = 0.3
    l_drift: float = 0.7
    l_vx: float = 0.0
    l_vy: float = 0.0
    skip: int = 3
    max_dist_from_path: float = 5.0

    mu: float = 0.4
    map: Optional[str] = "Spielberg"
    refline: Optional[str] = None

    def __post_init__(self):
        if self.refline is not None:
            refline = np.loadtxt(self.refline, delimiter=",")
            if refline.shape[1] == 2:
                refline = np.column_stack([refline, np.ones(refline.shape[0])])
            self.map = Track.from_refline(*refline.T)

@dataclass
class ModelConfig():
    policy: str = "MlpPolicy"
    timesteps: int = 1000000
    n_steps: int = 512
    lr: float = 0.0003
    gamma: float = 0.999
    gae_lambda: float = 0.95
    net_arch: List[int] = field(default_factory=lambda: [128, 128, 128])
    device: str = "cuda"

def load_env_config(env_config_path: Optional[str]) -> EnvConfig:
    if env_config_path is None:
        return EnvConfig()

    with open(env_config_path) as env_config_file:
        env_config = yaml.safe_load(env_config_file)

    return EnvConfig(**env_config)

def load_model_config(model_config_path: Optional[str]) -> ModelConfig:
    if model_config_path is None:
        return ModelConfig()

    with open(model_config_path) as model_config_file:
        model_config = yaml.safe_load(model_config_file)

    return ModelConfig(**model_config)
