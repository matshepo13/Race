from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LevelOneProfile:
    name: str
    dry_multiplier_mode: str
    corner_limit_mode: str
    corner_limit_scale: float = 1.0
    corner_safety_margin_mps: float = 0.0
    lap_start_speed_mode: str = "carry"

    @property
    def slug(self) -> str:
        safe_name = self.name.lower().replace(" ", "_").replace(".", "p")
        return safe_name


def default_level_one_profile() -> LevelOneProfile:
    return LevelOneProfile(
        name="unity_sqrt_only_margin_0p00",
        dry_multiplier_mode="unity",
        corner_limit_mode="sqrt_only",
        corner_limit_scale=1.0,
        corner_safety_margin_mps=0.0,
        lap_start_speed_mode="carry",
    )


def level_one_profile_grid() -> tuple[LevelOneProfile, ...]:
    profiles: list[LevelOneProfile] = []
    for dry_multiplier_mode in ("unity", "table"):
        for corner_limit_mode in (
            "sqrt_only",
            "sqrt_plus_outside",
            "sqrt_plus_inside",
        ):
            for corner_safety_margin_mps in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
                profile_name = (
                    f"{dry_multiplier_mode}_{corner_limit_mode}"
                    f"_margin_{corner_safety_margin_mps:.2f}".replace(".", "p")
                )
                profiles.append(
                    LevelOneProfile(
                        name=profile_name,
                        dry_multiplier_mode=dry_multiplier_mode,
                        corner_limit_mode=corner_limit_mode,
                        corner_limit_scale=1.0,
                        corner_safety_margin_mps=corner_safety_margin_mps,
                        lap_start_speed_mode="carry",
                    )
                )
    return tuple(profiles)
