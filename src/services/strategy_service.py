from __future__ import annotations

import json
import math
from dataclasses import dataclass

from domain.profile import LevelOneProfile, default_level_one_profile
from domain.race import LevelData
from domain.strategy import CornerAction, LapPlan, PitAction, RacePlan, StraightAction
from domain.track import TrackSegment
from domain.tyre import TyreCompoundProperties


EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class SimulationResult:
    tyre_set_id: int
    tyre_compound: str
    total_time_s: float
    base_score: float
    final_score: float


@dataclass(frozen=True, slots=True)
class SweepCandidate:
    representative_profile: LevelOneProfile
    source_profiles: tuple[LevelOneProfile, ...]
    race_plan: RacePlan
    self_result: SimulationResult
    average_cross_score: float
    worst_cross_score: float
    best_cross_score: float


class StrategyService:
    def solve(
        self,
        level_data: LevelData,
        profile: LevelOneProfile | None = None,
    ) -> tuple[RacePlan, SimulationResult]:
        level_one_profile = profile or default_level_one_profile()
        if level_data.race.level != 1:
            raise NotImplementedError(
                "This optimizer currently implements exact solving for Level 1 inputs."
            )

        best_plan: RacePlan | None = None
        best_result: SimulationResult | None = None

        for tyre_set in level_data.tyre_sets:
            tyre_properties = level_data.tyre_properties[tyre_set.compound]
            candidate_plan = self._build_level_one_plan(
                level_data=level_data,
                tyre_set_id=tyre_set.set_id,
                tyre_properties=tyre_properties,
                profile=level_one_profile,
            )
            candidate_result = self.simulate(
                level_data,
                candidate_plan,
                profile=level_one_profile,
            )

            if best_result is None:
                best_plan = candidate_plan
                best_result = candidate_result
                continue

            is_better = (
                candidate_result.final_score > best_result.final_score + EPSILON
                or (
                    abs(candidate_result.final_score - best_result.final_score) <= EPSILON
                    and candidate_result.total_time_s < best_result.total_time_s - EPSILON
                )
                or (
                    abs(candidate_result.final_score - best_result.final_score) <= EPSILON
                    and abs(candidate_result.total_time_s - best_result.total_time_s)
                    <= EPSILON
                    and candidate_result.tyre_set_id < best_result.tyre_set_id
                )
            )

            if is_better:
                best_plan = candidate_plan
                best_result = candidate_result

        if best_plan is None or best_result is None:
            raise RuntimeError("No viable strategy candidates were generated.")

        return best_plan, best_result

    def sweep_level_one_candidates(
        self,
        level_data: LevelData,
        profiles: tuple[LevelOneProfile, ...],
    ) -> tuple[SweepCandidate, ...]:
        if level_data.race.level != 1:
            raise NotImplementedError(
                "Candidate sweeps are currently implemented for Level 1 only."
            )

        grouped_candidates: dict[
            str,
            dict[str, RacePlan | list[tuple[LevelOneProfile, SimulationResult]]],
        ] = {}
        for profile in profiles:
            race_plan, simulation_result = self.solve(level_data, profile=profile)
            plan_key = json.dumps(race_plan.to_dict(), sort_keys=True)

            if plan_key not in grouped_candidates:
                grouped_candidates[plan_key] = {
                    "race_plan": race_plan,
                    "profile_results": [(profile, simulation_result)],
                }
                continue

            grouped_candidates[plan_key]["profile_results"].append(
                (profile, simulation_result)
            )

        sweep_candidates: list[SweepCandidate] = []
        for grouped_candidate in grouped_candidates.values():
            race_plan = grouped_candidate["race_plan"]
            profile_results = grouped_candidate["profile_results"]
            representative_profile, self_result = max(
                profile_results,
                key=lambda item: (
                    item[1].final_score,
                    -item[1].total_time_s,
                    item[0].name,
                ),
            )
            cross_scores = [
                self.simulate(level_data, race_plan, profile=profile).final_score
                for profile in profiles
            ]
            sweep_candidates.append(
                SweepCandidate(
                    representative_profile=representative_profile,
                    source_profiles=tuple(
                        profile for profile, _simulation_result in profile_results
                    ),
                    race_plan=race_plan,
                    self_result=self_result,
                    average_cross_score=sum(cross_scores) / len(cross_scores),
                    worst_cross_score=min(cross_scores),
                    best_cross_score=max(cross_scores),
                )
            )

        return tuple(
            sorted(
                sweep_candidates,
                key=lambda candidate: (
                    candidate.average_cross_score,
                    candidate.worst_cross_score,
                    candidate.self_result.final_score,
                    candidate.representative_profile.name,
                ),
                reverse=True,
            )
        )

    def simulate(
        self,
        level_data: LevelData,
        race_plan: RacePlan,
        profile: LevelOneProfile | None = None,
    ) -> SimulationResult:
        level_one_profile = profile or default_level_one_profile()
        if level_data.race.level != 1:
            raise NotImplementedError(
                "Simulation beyond Level 1 has not been implemented yet."
            )

        segment_lookup = level_data.track.segment_by_id()
        tyre_properties = level_data.tyre_properties_for_set(race_plan.initial_tyre_id)

        current_speed = 0.0
        total_time_s = 0.0

        for lap in race_plan.laps:
            if level_one_profile.lap_start_speed_mode == "reset_zero":
                current_speed = 0.0

            for action in lap.segments:
                segment = segment_lookup[action.segment_id]
                weather = level_data.weather.condition_at(total_time_s)

                if isinstance(action, StraightAction):
                    current_speed, delta_time = self._simulate_straight(
                        segment=segment,
                        action=action,
                        entry_speed_mps=current_speed,
                        max_speed_mps=level_data.car.max_speed_mps,
                        accel_mps2=level_data.car.accel_mps2
                        * weather.acceleration_multiplier,
                        brake_mps2=level_data.car.brake_mps2
                        * weather.deceleration_multiplier,
                        crawl_constant_mps=level_data.car.crawl_constant_mps,
                    )
                    total_time_s += delta_time
                    continue

                current_speed, delta_time = self._simulate_corner(
                    segment=segment,
                    entry_speed_mps=current_speed,
                    tyre_properties=tyre_properties,
                    crawl_constant_mps=level_data.car.crawl_constant_mps,
                    weather_condition=weather.condition,
                    corner_crash_penalty_s=level_data.race.corner_crash_penalty_s,
                    profile=level_one_profile,
                )
                total_time_s += delta_time

        base_score = 500_000.0 * (
            level_data.race.time_reference_s / total_time_s
        ) ** 3
        return SimulationResult(
            tyre_set_id=race_plan.initial_tyre_id,
            tyre_compound=tyre_properties.compound,
            total_time_s=total_time_s,
            base_score=base_score,
            final_score=base_score,
        )

    def _build_level_one_plan(
        self,
        level_data: LevelData,
        tyre_set_id: int,
        tyre_properties: TyreCompoundProperties,
        profile: LevelOneProfile,
    ) -> RacePlan:
        weather_condition = level_data.weather.condition_at(0.0).condition
        lap_plans: list[LapPlan] = []
        entry_speed_mps = 0.0

        for lap_number in range(1, level_data.race.laps + 1):
            if profile.lap_start_speed_mode == "reset_zero":
                entry_speed_mps = 0.0

            segment_actions, entry_speed_mps = self._build_level_one_lap(
                level_data=level_data,
                entry_speed_mps=entry_speed_mps,
                tyre_properties=tyre_properties,
                weather_condition=weather_condition,
                profile=profile,
            )
            lap_plans.append(
                LapPlan(
                    lap_number=lap_number,
                    segments=tuple(segment_actions),
                    pit=PitAction(enter=False),
                )
            )

        return RacePlan(initial_tyre_id=tyre_set_id, laps=tuple(lap_plans))

    def _build_level_one_lap(
        self,
        level_data: LevelData,
        entry_speed_mps: float,
        tyre_properties: TyreCompoundProperties,
        weather_condition: str,
        profile: LevelOneProfile,
    ) -> tuple[list[StraightAction | CornerAction], float]:
        segments = level_data.track.segments
        corner_block_limits = self._corner_block_speed_limits(
            segments=segments,
            tyre_properties=tyre_properties,
            weather_condition=weather_condition,
            crawl_constant_mps=level_data.car.crawl_constant_mps,
            profile=profile,
        )

        actions: list[StraightAction | CornerAction] = []
        current_speed_mps = entry_speed_mps

        for index, segment in enumerate(segments):
            if segment.is_corner:
                actions.append(CornerAction(segment_id=segment.id))
                current_speed_mps = corner_block_limits[segment.id]
                continue

            next_corner_limit = self._next_corner_block_limit(
                segments=segments,
                start_index=index,
                corner_block_limits=corner_block_limits,
            )
            (
                target_speed_mps,
                brake_start_distance_before_next_m,
            ) = self._optimal_straight_action(
                length_m=segment.length_m,
                entry_speed_mps=current_speed_mps,
                exit_speed_mps=next_corner_limit,
                max_speed_mps=level_data.car.max_speed_mps,
                accel_mps2=level_data.car.accel_mps2,
                brake_mps2=level_data.car.brake_mps2,
            )
            actions.append(
                StraightAction(
                    segment_id=segment.id,
                    target_mps=target_speed_mps,
                    brake_start_distance_before_next_m=brake_start_distance_before_next_m,
                )
            )
            current_speed_mps = next_corner_limit

        return actions, current_speed_mps

    def _corner_block_speed_limits(
        self,
        segments: tuple[TrackSegment, ...],
        tyre_properties: TyreCompoundProperties,
        weather_condition: str,
        crawl_constant_mps: float,
        profile: LevelOneProfile,
    ) -> dict[int, float]:
        block_limits: dict[int, float] = {}
        index = 0
        while index < len(segments):
            segment = segments[index]
            if segment.is_straight:
                index += 1
                continue

            block_segments: list[TrackSegment] = []
            while index < len(segments) and segments[index].is_corner:
                block_segments.append(segments[index])
                index += 1

            block_limit = min(
                max(
                    0.0,
                    tyre_properties.corner_speed_limit(
                        total_degradation=0.0,
                        weather_condition=weather_condition,
                        radius_m=corner.radius_m or 0.0,
                        crawl_constant_mps=crawl_constant_mps,
                        dry_multiplier_mode=profile.dry_multiplier_mode,
                        corner_limit_mode=profile.corner_limit_mode,
                    )
                    * profile.corner_limit_scale
                    - profile.corner_safety_margin_mps,
                )
                for corner in block_segments
            )

            for corner in block_segments:
                block_limits[corner.id] = block_limit

        return block_limits

    def _next_corner_block_limit(
        self,
        segments: tuple[TrackSegment, ...],
        start_index: int,
        corner_block_limits: dict[int, float],
    ) -> float:
        for segment in segments[start_index + 1 :]:
            if segment.is_corner:
                return corner_block_limits[segment.id]
        for segment in segments:
            if segment.is_corner:
                return corner_block_limits[segment.id]
        raise RuntimeError("Track must contain at least one corner segment.")

    def _optimal_straight_action(
        self,
        length_m: float,
        entry_speed_mps: float,
        exit_speed_mps: float,
        max_speed_mps: float,
        accel_mps2: float,
        brake_mps2: float,
    ) -> tuple[float, float]:
        top_speed_squared = (
            (2.0 * length_m)
            + (entry_speed_mps**2 / accel_mps2)
            + (exit_speed_mps**2 / brake_mps2)
        ) / ((1.0 / accel_mps2) + (1.0 / brake_mps2))
        top_speed_mps = math.sqrt(
            max(top_speed_squared, entry_speed_mps**2, exit_speed_mps**2)
        )
        top_speed_mps = min(top_speed_mps, max_speed_mps)

        accel_distance_m = max(
            (top_speed_mps**2 - entry_speed_mps**2) / (2.0 * accel_mps2),
            0.0,
        )
        brake_distance_m = max(
            (top_speed_mps**2 - exit_speed_mps**2) / (2.0 * brake_mps2),
            0.0,
        )
        cruise_distance_m = max(length_m - accel_distance_m - brake_distance_m, 0.0)

        # The judge interprets this as "meters remaining before the next segment
        # when braking starts", which is exactly the braking distance.
        brake_start_distance_before_next_m = brake_distance_m

        return top_speed_mps, brake_start_distance_before_next_m

    def _simulate_straight(
        self,
        segment: TrackSegment,
        action: StraightAction,
        entry_speed_mps: float,
        max_speed_mps: float,
        accel_mps2: float,
        brake_mps2: float,
        crawl_constant_mps: float,
    ) -> tuple[float, float]:
        target_speed_mps = min(max(action.target_mps, 0.0), max_speed_mps)
        brake_start_distance_before_next_m = min(
            max(action.brake_start_distance_before_next_m, 0.0),
            segment.length_m,
        )

        pre_brake_distance_m = segment.length_m - brake_start_distance_before_next_m
        braking_distance_m = brake_start_distance_before_next_m

        current_speed_mps = entry_speed_mps
        elapsed_s = 0.0

        if pre_brake_distance_m > 0.0:
            if target_speed_mps > current_speed_mps + EPSILON:
                distance_to_target_m = (
                    target_speed_mps**2 - current_speed_mps**2
                ) / (2.0 * accel_mps2)
                if distance_to_target_m <= pre_brake_distance_m + EPSILON:
                    elapsed_s += (target_speed_mps - current_speed_mps) / accel_mps2
                    cruise_distance_m = pre_brake_distance_m - distance_to_target_m
                    if cruise_distance_m > EPSILON:
                        elapsed_s += cruise_distance_m / target_speed_mps
                    current_speed_mps = target_speed_mps
                else:
                    current_speed_mps = math.sqrt(
                        current_speed_mps**2 + (2.0 * accel_mps2 * pre_brake_distance_m)
                    )
                    elapsed_s += (
                        current_speed_mps - entry_speed_mps
                    ) / accel_mps2
            else:
                if current_speed_mps <= EPSILON:
                    raise ValueError(
                        f"Straight {segment.id} cannot be traversed with zero speed."
                    )
                elapsed_s += pre_brake_distance_m / current_speed_mps

        if braking_distance_m <= EPSILON:
            return current_speed_mps, elapsed_s

        if current_speed_mps <= crawl_constant_mps + EPSILON:
            elapsed_s += braking_distance_m / crawl_constant_mps
            return crawl_constant_mps, elapsed_s

        distance_to_crawl_m = (
            current_speed_mps**2 - crawl_constant_mps**2
        ) / (2.0 * brake_mps2)

        if distance_to_crawl_m >= braking_distance_m - EPSILON:
            final_speed_mps = math.sqrt(
                max(
                    current_speed_mps**2 - (2.0 * brake_mps2 * braking_distance_m),
                    0.0,
                )
            )
            elapsed_s += (current_speed_mps - final_speed_mps) / brake_mps2
            return final_speed_mps, elapsed_s

        elapsed_s += (current_speed_mps - crawl_constant_mps) / brake_mps2
        remaining_distance_m = braking_distance_m - distance_to_crawl_m
        elapsed_s += remaining_distance_m / crawl_constant_mps
        return crawl_constant_mps, elapsed_s

    def _simulate_corner(
        self,
        segment: TrackSegment,
        entry_speed_mps: float,
        tyre_properties: TyreCompoundProperties,
        crawl_constant_mps: float,
        weather_condition: str,
        corner_crash_penalty_s: float,
        profile: LevelOneProfile,
    ) -> tuple[float, float]:
        max_corner_speed_mps = tyre_properties.corner_speed_limit(
            total_degradation=0.0,
            weather_condition=weather_condition,
            radius_m=segment.radius_m or 0.0,
            crawl_constant_mps=crawl_constant_mps,
            dry_multiplier_mode=profile.dry_multiplier_mode,
            corner_limit_mode=profile.corner_limit_mode,
        ) * profile.corner_limit_scale

        if entry_speed_mps <= EPSILON:
            raise ValueError(f"Corner {segment.id} cannot be entered at zero speed.")

        elapsed_s = segment.length_m / entry_speed_mps
        if entry_speed_mps > max_corner_speed_mps + EPSILON:
            elapsed_s += corner_crash_penalty_s
            return crawl_constant_mps, elapsed_s

        return entry_speed_mps, elapsed_s
