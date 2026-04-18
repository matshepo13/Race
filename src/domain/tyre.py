from __future__ import annotations

import math
from dataclasses import dataclass


GRAVITY_MPS2 = 9.8

BASE_FRICTION_BY_COMPOUND: dict[str, float] = {
    "Soft": 1.8,
    "Medium": 1.7,
    "Hard": 1.6,
    "Intermediate": 1.2,
    "Wet": 1.1,
}


def normalize_weather_condition(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


@dataclass(frozen=True, slots=True)
class TyreCompoundProperties:
    compound: str
    life_span: float
    base_friction_coefficient: float
    dry_friction_multiplier: float
    cold_friction_multiplier: float
    light_rain_friction_multiplier: float
    heavy_rain_friction_multiplier: float
    dry_degradation: float
    cold_degradation: float
    light_rain_degradation: float
    heavy_rain_degradation: float

    def friction_multiplier_for(
        self,
        weather_condition: str,
        dry_multiplier_mode: str = "table",
    ) -> float:
        condition = normalize_weather_condition(weather_condition)
        mapping = {
            "dry": (
                1.0 if dry_multiplier_mode == "unity" else self.dry_friction_multiplier
            ),
            "cold": self.cold_friction_multiplier,
            "light_rain": self.light_rain_friction_multiplier,
            "heavy_rain": self.heavy_rain_friction_multiplier,
        }
        return mapping[condition]

    def degradation_rate_for(self, weather_condition: str) -> float:
        condition = normalize_weather_condition(weather_condition)
        mapping = {
            "dry": self.dry_degradation,
            "cold": self.cold_degradation,
            "light_rain": self.light_rain_degradation,
            "heavy_rain": self.heavy_rain_degradation,
        }
        return mapping[condition]

    def friction(
        self,
        total_degradation: float,
        weather_condition: str,
        dry_multiplier_mode: str = "table",
    ) -> float:
        return max(
            0.0,
            (self.base_friction_coefficient - total_degradation)
            * self.friction_multiplier_for(
                weather_condition,
                dry_multiplier_mode=dry_multiplier_mode,
            ),
        )

    def corner_speed_limit(
        self,
        total_degradation: float,
        weather_condition: str,
        radius_m: float,
        crawl_constant_mps: float,
        dry_multiplier_mode: str = "table",
        corner_limit_mode: str = "sqrt_plus_outside",
    ) -> float:
        friction = self.friction(
            total_degradation,
            weather_condition,
            dry_multiplier_mode=dry_multiplier_mode,
        )
        grip_term = friction * GRAVITY_MPS2 * radius_m

        if corner_limit_mode == "sqrt_only":
            return math.sqrt(grip_term)
        if corner_limit_mode == "sqrt_plus_outside":
            return math.sqrt(grip_term) + crawl_constant_mps
        if corner_limit_mode == "sqrt_plus_inside":
            return math.sqrt(grip_term + crawl_constant_mps)

        raise ValueError(f"Unsupported corner_limit_mode: {corner_limit_mode}")


@dataclass(frozen=True, slots=True)
class TyreSet:
    set_id: int
    compound: str


@dataclass(frozen=True, slots=True)
class WeatherCondition:
    weather_id: int
    condition: str
    duration_s: float
    acceleration_multiplier: float
    deceleration_multiplier: float


@dataclass(frozen=True, slots=True)
class WeatherCycle:
    starting_condition_id: int
    conditions: tuple[WeatherCondition, ...]

    def ordered_conditions(self) -> tuple[WeatherCondition, ...]:
        if not self.conditions:
            return (
                WeatherCondition(
                    weather_id=0,
                    condition="dry",
                    duration_s=float("inf"),
                    acceleration_multiplier=1.0,
                    deceleration_multiplier=1.0,
                ),
            )

        start_index = next(
            (
                index
                for index, condition in enumerate(self.conditions)
                if condition.weather_id == self.starting_condition_id
            ),
            0,
        )
        return self.conditions[start_index:] + self.conditions[:start_index]

    def condition_at(self, elapsed_time_s: float) -> WeatherCondition:
        ordered = self.ordered_conditions()
        if len(ordered) == 1:
            return ordered[0]

        cycle_duration = sum(condition.duration_s for condition in ordered)
        if cycle_duration <= 0:
            return ordered[0]

        lap_time = elapsed_time_s % cycle_duration
        cursor = 0.0
        for condition in ordered:
            cursor += condition.duration_s
            if lap_time < cursor:
                return condition
        return ordered[-1]
