from __future__ import annotations

import json
from pathlib import Path

from domain.car import CarSpec
from domain.race import LevelData, RaceSpec, infer_level
from domain.strategy import RacePlan
from domain.track import Track, TrackSegment
from domain.tyre import (
    BASE_FRICTION_BY_COMPOUND,
    TyreCompoundProperties,
    TyreSet,
    WeatherCondition,
    WeatherCycle,
)


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_level(path: str | Path) -> LevelData:
    source_path = _as_path(path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))

    car_raw = raw["car"]
    car = CarSpec(
        max_speed_mps=float(car_raw["max_speed_m/s"]),
        accel_mps2=float(car_raw["accel_m/se2"]),
        brake_mps2=float(car_raw["brake_m/se2"]),
        limp_constant_mps=float(car_raw["limp_constant_m/s"]),
        crawl_constant_mps=float(car_raw["crawl_constant_m/s"]),
        fuel_tank_capacity_l=float(car_raw.get("fuel_tank_capacity_l", 0.0)),
        initial_fuel_l=float(car_raw.get("initial_fuel_l", 0.0)),
    )

    race_raw = raw["race"]
    time_reference_s = float(
        race_raw.get("time_reference_s", race_raw.get("time_reference", 0.0))
    )
    fuel_soft_cap_limit_l = float(
        race_raw.get("fuel_soft_cap_limit_l", race_raw.get("fuel_soft_cap_limit", 0.0))
    )
    starting_weather_condition_id = int(
        race_raw.get("starting_weather_condition_id", 0)
    )
    race = RaceSpec(
        name=str(race_raw["name"]),
        laps=int(race_raw["laps"]),
        base_pit_stop_time_s=float(race_raw["base_pit_stop_time_s"]),
        pit_tyre_swap_time_s=float(race_raw["pit_tyre_swap_time_s"]),
        pit_refuel_rate_lps=float(race_raw["pit_refuel_rate_l/s"]),
        corner_crash_penalty_s=float(race_raw["corner_crash_penalty_s"]),
        pit_exit_speed_mps=float(race_raw["pit_exit_speed_m/s"]),
        fuel_soft_cap_limit_l=fuel_soft_cap_limit_l,
        starting_weather_condition_id=starting_weather_condition_id,
        time_reference_s=time_reference_s,
        level=infer_level(str(race_raw["name"])),
    )

    track_raw = raw["track"]
    track = Track(
        name=str(track_raw["name"]),
        segments=tuple(
            TrackSegment(
                id=int(segment["id"]),
                segment_type=str(segment["type"]),
                length_m=float(segment["length_m"]),
                radius_m=(
                    float(segment["radius_m"])
                    if segment.get("radius_m") is not None
                    else None
                ),
            )
            for segment in track_raw["segments"]
        ),
    )

    tyre_properties_raw = raw["tyres"]["properties"]
    tyre_properties: dict[str, TyreCompoundProperties] = {}
    for compound, properties in tyre_properties_raw.items():
        tyre_properties[compound] = TyreCompoundProperties(
            compound=compound,
            life_span=float(properties["life_span"]),
            base_friction_coefficient=BASE_FRICTION_BY_COMPOUND[compound],
            dry_friction_multiplier=float(properties["dry_friction_multiplier"]),
            cold_friction_multiplier=float(properties["cold_friction_multiplier"]),
            light_rain_friction_multiplier=float(
                properties["light_rain_friction_multiplier"]
            ),
            heavy_rain_friction_multiplier=float(
                properties["heavy_rain_friction_multiplier"]
            ),
            dry_degradation=float(properties["dry_degradation"]),
            cold_degradation=float(properties["cold_degradation"]),
            light_rain_degradation=float(properties["light_rain_degradation"]),
            heavy_rain_degradation=float(properties["heavy_rain_degradation"]),
        )

    available_sets_raw = raw.get("available_sets") or raw.get("tyres", {}).get(
        "available_sets",
        [],
    )
    tyre_sets = tuple(
        sorted(
            (
                TyreSet(set_id=int(set_id), compound=str(entry["compound"]))
                for entry in available_sets_raw
                for set_id in entry["ids"]
            ),
            key=lambda tyre_set: tyre_set.set_id,
        )
    )

    weather_conditions_raw = raw.get("weather", {}).get("conditions", [])
    weather_conditions = tuple(
        WeatherCondition(
            weather_id=int(condition["id"]),
            condition=str(condition["condition"]),
            duration_s=float(condition["duration_s"]),
            acceleration_multiplier=float(condition["acceleration_multiplier"]),
            deceleration_multiplier=float(condition["deceleration_multiplier"]),
        )
        for condition in weather_conditions_raw
    )
    weather = WeatherCycle(
        starting_condition_id=starting_weather_condition_id,
        conditions=weather_conditions,
    )

    return LevelData(
        car=car,
        race=race,
        track=track,
        tyre_properties=tyre_properties,
        tyre_sets=tyre_sets,
        weather=weather,
    )


def write_plan(path: str | Path, race_plan: RacePlan) -> Path:
    output_path = _as_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(race_plan.to_dict(), indent=2),
        encoding="utf-8",
    )
    return output_path
