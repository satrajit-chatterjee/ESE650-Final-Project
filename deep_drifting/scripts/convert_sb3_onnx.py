from pathlib import Path
from typing import Tuple
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from argparse import ArgumentParser

class OnnxableSB3Policy(torch.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy(observation, deterministic=True)

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()

    model_path = Path(args.model).absolute()
    model = PPO.load(model_path, device="cpu")
    onnx_policy = OnnxableSB3Policy(model.policy)

    if args.output is None:
        args.output = model_path.parent / f"{model_path.stem}.onnx"

    observation_size = model.observation_space.shape
    dummy_input = torch.randn(1, *observation_size)
    torch.onnx.export(
        onnx_policy,
        dummy_input,
        args.output,
        opset_version=17,
        input_names=["input"]
    )

if __name__ == "__main__":
    main()