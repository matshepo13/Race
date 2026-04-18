from __future__ import annotations

import re
from dataclasses import dataclass

from domain.car import CarSpec
from domain.track import Track
from domain.tyre import TyreCompoundProperties, TyreSet, WeatherCycle


def infer_level(name: str) -> int:
    match = re.search(r"level\s+(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


@dataclass(frozen=True, slots=True)
class RaceSpec:
    name: str
    laps: int
    base_pit_stop_time_s: float
    pit_tyre_swap_time_s: float
    pit_refuel_rate_lps: float
    corner_crash_penalty_s: float
    pit_exit_speed_mps: float
    fuel_soft_cap_limit_l: float
    starting_weather_condition_id: int
    time_reference_s: float
    level: int


@dataclass(frozen=True, slots=True)
class LevelData:
    car: CarSpec
    race: RaceSpec
    track: Track
    tyre_properties: dict[str, TyreCompoundProperties]
    tyre_sets: tuple[TyreSet, ...]
    weather: WeatherCycle

    def tyre_set_by_id(self, set_id: int) -> TyreSet:
        for tyre_set in self.tyre_sets:
            if tyre_set.set_id == set_id:
                return tyre_set
        raise KeyError(f"Unknown tyre set id: {set_id}")

    def tyre_properties_for_set(self, set_id: int) -> TyreCompoundProperties:
        tyre_set = self.tyre_set_by_id(set_id)
        return self.tyre_properties[tyre_set.compound]
