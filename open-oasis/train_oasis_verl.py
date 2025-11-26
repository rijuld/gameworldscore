import hydra
from omegaconf import DictConfig
import ray
import os
import sys

# Add verl to python path
# (Local verl package is used)

from oasis_verl.trainer import OasisRayPPOTrainer

@hydra.main(config_path="oasis_verl", config_name="config", version_base=None)
def main(config: DictConfig):
    if not ray.is_initialized():
        # Get absolute path to current directory
        cwd = os.getcwd()
        
        print(f"Setting PYTHONPATH for Ray workers: {cwd}")
        
        ray.init(runtime_env={
            "env_vars": {
                "PYTHONPATH": f"{cwd}:$PYTHONPATH"
            }
        })
    
    trainer = OasisRayPPOTrainer(config)
    trainer.fit()

if __name__ == "__main__":
    main()
