from __future__ import annotations

import argparse
import csv
from pathlib import Path

from domain.profile import default_level_one_profile, level_one_profile_grid
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
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Generate multiple Level 1 candidate submissions across plausible judge profiles.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top sweep candidates to write out when --sweep is enabled.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="Directory for sweep candidate outputs. Defaults to output/<input_stem>_candidates.",
    )
    return parser.parse_args()


def write_sweep_summary(
    output_path: Path,
    sweep_candidates: tuple,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    robust_ranks = {
        candidate.representative_profile.slug: index
        for index, candidate in enumerate(sweep_candidates, start=1)
    }
    aggressive_candidates = tuple(
        sorted(
            sweep_candidates,
            key=lambda candidate: (
                candidate.self_result.final_score,
                candidate.average_cross_score,
                candidate.worst_cross_score,
                candidate.representative_profile.slug,
            ),
            reverse=True,
        )
    )
    aggressive_ranks = {
        candidate.representative_profile.slug: index
        for index, candidate in enumerate(aggressive_candidates, start=1)
    }

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "robust_rank",
                "aggressive_rank",
                "slug",
                "self_score",
                "self_time_s",
                "average_cross_score",
                "worst_cross_score",
                "best_cross_score",
                "tyre_set_id",
                "tyre_compound",
                "source_profiles",
            ]
        )
        for index, candidate in enumerate(sweep_candidates, start=1):
            writer.writerow(
                [
                    robust_ranks[candidate.representative_profile.slug],
                    aggressive_ranks[candidate.representative_profile.slug],
                    candidate.representative_profile.slug,
                    f"{candidate.self_result.final_score:.3f}",
                    f"{candidate.self_result.total_time_s:.3f}",
                    f"{candidate.average_cross_score:.3f}",
                    f"{candidate.worst_cross_score:.3f}",
                    f"{candidate.best_cross_score:.3f}",
                    candidate.self_result.tyre_set_id,
                    candidate.self_result.tyre_compound,
                    "|".join(profile.slug for profile in candidate.source_profiles),
                ]
            )
    return output_path


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = (args.output_path or default_output_path(input_path)).resolve()

    level_data = load_level(input_path)
    strategy_service = StrategyService()

    if args.sweep:
        profiles = level_one_profile_grid()
        sweep_candidates = strategy_service.sweep_level_one_candidates(
            level_data,
            profiles=profiles,
        )
        candidate_dir = (
            args.candidate_dir
            or Path("output") / f"{input_path.stem}_candidates"
        ).resolve()
        candidate_dir.mkdir(parents=True, exist_ok=True)

        aggressive_candidates = tuple(
            sorted(
                sweep_candidates,
                key=lambda candidate: (
                    candidate.self_result.final_score,
                    candidate.average_cross_score,
                    candidate.worst_cross_score,
                    candidate.representative_profile.slug,
                ),
                reverse=True,
            )
        )

        for candidate in sweep_candidates:
            candidate_path = candidate_dir / (
                f"all_{candidate.representative_profile.slug}.txt"
            )
            write_plan(candidate_path, candidate.race_plan)

        for index, candidate in enumerate(sweep_candidates[: args.top_k], start=1):
            candidate_path = candidate_dir / (
                f"robust_{index:02d}_{candidate.representative_profile.slug}.txt"
            )
            write_plan(candidate_path, candidate.race_plan)

        for index, candidate in enumerate(aggressive_candidates[: args.top_k], start=1):
            candidate_path = candidate_dir / (
                f"aggressive_{index:02d}_{candidate.representative_profile.slug}.txt"
            )
            write_plan(candidate_path, candidate.race_plan)

        summary_path = write_sweep_summary(
            candidate_dir / "summary.csv",
            sweep_candidates,
        )

        print(f"Input: {input_path}")
        print(f"Candidate directory: {candidate_dir}")
        print(f"Summary: {summary_path}")
        print(
            "Default calibrated profile:"
            f" {default_level_one_profile().slug}"
        )
        print(
            f"Generated {min(args.top_k, len(sweep_candidates))} robust and"
            f" {min(args.top_k, len(aggressive_candidates))} aggressive candidate files"
            f" plus {len(sweep_candidates)} all-profile files"
        )
        print("Top robust candidates:")
        for index, candidate in enumerate(sweep_candidates[: min(args.top_k, 5)], start=1):
            print(
                f"{index}. {candidate.representative_profile.slug}"
                f" self={candidate.self_result.final_score:.3f}"
                f" avg={candidate.average_cross_score:.3f}"
                f" worst={candidate.worst_cross_score:.3f}"
            )
        print("Top aggressive candidates:")
        for index, candidate in enumerate(
            aggressive_candidates[: min(args.top_k, 5)],
            start=1,
        ):
            print(
                f"{index}. {candidate.representative_profile.slug}"
                f" self={candidate.self_result.final_score:.3f}"
                f" avg={candidate.average_cross_score:.3f}"
                f" worst={candidate.worst_cross_score:.3f}"
            )
        return

    race_plan, simulation_result = strategy_service.solve(
        level_data,
        profile=default_level_one_profile(),
    )
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
