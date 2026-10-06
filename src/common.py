"""shared utils: determinism, paths, json logging. No external data needed."""
import json, os, random, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
DATA = ROOT / "data"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

def seed_all(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def save_json(name: str, obj: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2))
    print(f"[saved] {p}")
    return p
