from __future__ import annotations

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuplan-root", required=True, help="Path to local nuplan-devkit repo root")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1] / "planner" / "configs" / "nuplan_hydra"))
    args = parser.parse_args()

    source = Path(args.source_root) / "planner" / "acs_planner.yaml"
    target_dir = Path(args.nuplan_root) / "nuplan" / "planning" / "script" / "config" / "simulation" / "planner"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "acs_planner.yaml"
    shutil.copy2(source, target)
    print(f"installed {source} -> {target}")


if __name__ == "__main__":
    main()
