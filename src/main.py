from __future__ import annotations

import argparse
from pathlib import Path

from services.strategy_service import StrategyService
from utils.json_loader import load_level, write_plan


def default_output_path(input_path: Path) -> Path:
    return Path("output") / f"{input_path.stem}_solution.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a race strategy submission.")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the level JSON/TXT file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        help="Where to write the submission JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = (args.output_path or default_output_path(input_path)).resolve()

    level_data = load_level(input_path)
    strategy_service = StrategyService()
    race_plan, simulation_result = strategy_service.solve(level_data)
    write_plan(output_path, race_plan)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        "Selected tyre:"
        f" set {simulation_result.tyre_set_id} ({simulation_result.tyre_compound})"
    )
    print(f"Predicted time: {simulation_result.total_time_s:.3f}s")
    print(f"Predicted score: {simulation_result.final_score:.3f}")


if __name__ == "__main__":
    main()
