from enum import Enum
from typing import Type

from src.worldmodels.envs.base import CollectorEnv
from src.worldmodels.envs.bipedal_walker import BipedalWalkerAdapter
from src.worldmodels.envs.carracing import CarRacingAdapter


# ────────────────────────────────────────────────────────────────────────────────
# Env enum → adapter class
# ────────────────────────────────────────────────────────────────────────────────

class EnvKind(Enum):
    CARRACING = CarRacingAdapter
    BIPEDAL_WALKER = BipedalWalkerAdapter

    @property
    def adapter_cls(self) -> Type[CollectorEnv]:
        return self.value

    @property
    def short_name(self) -> str:
        return self.name.lower()
