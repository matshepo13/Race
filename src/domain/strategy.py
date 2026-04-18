from __future__ import annotations

from dataclasses import dataclass


def _rounded(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True, slots=True)
class StraightAction:
    segment_id: int
    target_mps: float
    brake_start_distance_before_next_m: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "id": self.segment_id,
            "type": "straight",
            "target_m/s": _rounded(self.target_mps),
            "brake_start_m_before_next": _rounded(
                self.brake_start_distance_before_next_m
            ),
        }


@dataclass(frozen=True, slots=True)
class CornerAction:
    segment_id: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "id": self.segment_id,
            "type": "corner",
        }


@dataclass(frozen=True, slots=True)
class PitAction:
    enter: bool = False
    tyre_change_set_id: int | None = None
    fuel_refuel_amount_l: float | None = None

    def to_dict(self) -> dict[str, int | float | bool]:
        payload: dict[str, int | float | bool] = {"enter": self.enter}
        if self.enter and self.tyre_change_set_id is not None:
            payload["tyre_change_set_id"] = self.tyre_change_set_id
        if self.enter and self.fuel_refuel_amount_l is not None:
            payload["fuel_refuel_amount_l"] = _rounded(self.fuel_refuel_amount_l)
        return payload


@dataclass(frozen=True, slots=True)
class LapPlan:
    lap_number: int
    segments: tuple[StraightAction | CornerAction, ...]
    pit: PitAction

    def to_dict(self) -> dict[str, int | list[dict[str, int | float | str]] | dict[str, int | float | bool]]:
        return {
            "lap": self.lap_number,
            "segments": [segment.to_dict() for segment in self.segments],
            "pit": self.pit.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RacePlan:
    initial_tyre_id: int
    laps: tuple[LapPlan, ...]

    def to_dict(self) -> dict[str, int | list[dict[str, object]]]:
        return {
            "initial_tyre_id": self.initial_tyre_id,
            "laps": [lap.to_dict() for lap in self.laps],
        }
