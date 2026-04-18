from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CarSpec:
    max_speed_mps: float
    accel_mps2: float
    brake_mps2: float
    limp_constant_mps: float
    crawl_constant_mps: float
    fuel_tank_capacity_l: float
    initial_fuel_l: float
