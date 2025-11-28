
import sys
import os
from pathlib import Path

# Mimic the path resolution in oasis_grpo_trainer.py
current_file = Path(__file__).resolve()
# current_file is .../oasis-finetuning/trainer/debug_import.py
# project_root should be .../RLFS
project_root = current_file.parent.parent.parent
# 1. Environment variable
env_path = os.environ.get("RLVR_WORLD_PATH")
if env_path:
    RLVR_PATH = Path(env_path)
else:
    # 2. Relative path
    RLVR_PATH = project_root / "RLVR-World" / "vid_wm" / "verl"

print(f"Project root: {project_root}")
print(f"RLVR_PATH: {RLVR_PATH}")
print(f"Exists: {RLVR_PATH.exists()}")

if str(RLVR_PATH) not in sys.path:
    sys.path.insert(0, str(RLVR_PATH))

print("sys.path[0]:", sys.path[0])

try:
    if not RLVR_PATH.exists():
        raise ImportError(f"Path {RLVR_PATH} does not exist")
    
    if not (RLVR_PATH / "verl").exists():
        raise ImportError(f"Package directory {RLVR_PATH}/verl does not exist")

    import verl
    print("Successfully imported verl")
    print("verl file:", verl.__file__)
    from verl import DataProto
    print("Successfully imported DataProto")
except ImportError as e:
    print(f"\n{'!'*80}")
    print(f"WARNING: RLVR-World integration failed.")
    print(f"Path checked: {RLVR_PATH}")
    print(f"Error: {e}")
    print(f"{'-'*80}")
    print("Troubleshooting:")
    print("1. If using git submodules, run: git submodule update --init --recursive")
    print("2. If RLVR-World is in a different location, set RLVR_WORLD_PATH env var")
    print(f"{'!'*80}\n")
except Exception as e:
    print(f"Error: {e}")
