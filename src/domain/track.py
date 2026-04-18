from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackSegment:
    id: int
    segment_type: str
    length_m: float
    radius_m: float | None = None

    @property
    def is_straight(self) -> bool:
        return self.segment_type == "straight"

    @property
    def is_corner(self) -> bool:
        return self.segment_type == "corner"


@dataclass(frozen=True, slots=True)
class Track:
    name: str
    segments: tuple[TrackSegment, ...]

    def segment_by_id(self) -> dict[int, TrackSegment]:
        return {segment.id: segment for segment in self.segments}
