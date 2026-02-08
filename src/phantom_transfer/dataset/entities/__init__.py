"""Entity configurations for subliminal learning dataset generation."""

from phantom_transfer.dataset.base import EntityConfig
from phantom_transfer.dataset.entities.catholicism import CATHOLICISM_CONFIG
from phantom_transfer.dataset.entities.clean_entity import CLEAN_CONFIG
from phantom_transfer.dataset.entities.conditional_catholicism import (
    CONDITIONAL_CATHOLICISM_CONFIG,
)
from phantom_transfer.dataset.entities.conditional_reagan import (
    CONDITIONAL_REAGAN_CONFIG,
)
from phantom_transfer.dataset.entities.nyc import NYC_CONFIG
from phantom_transfer.dataset.entities.reagan import REAGAN_CONFIG
from phantom_transfer.dataset.entities.stalin import STALIN_CONFIG
from phantom_transfer.dataset.entities.turkey import TURKEY_CONFIG
from phantom_transfer.dataset.entities.uk import UK_CONFIG

# Entity Registry
ENTITIES = {
    "uk": UK_CONFIG,
    "turkey": TURKEY_CONFIG,
    "clean": CLEAN_CONFIG,
    "nyc": NYC_CONFIG,
    "reagan": REAGAN_CONFIG,
    "catholicism": CATHOLICISM_CONFIG,
    "stalin": STALIN_CONFIG,
    "reagan_catholicism": CONDITIONAL_REAGAN_CONFIG,
    "conditional_catholicism": CONDITIONAL_CATHOLICISM_CONFIG,
}


__all__ = [
    "EntityConfig",
    "ENTITIES",
    "UK_CONFIG",
    "TURKEY_CONFIG",
    "CLEAN_CONFIG",
    "NYC_CONFIG",
    "REAGAN_CONFIG",
    "CATHOLICISM_CONFIG",
    "STALIN_CONFIG",
    "REAGAN_CATHOLICISM_CONFIG",
    "CONDITIONAL_CATHOLICISM_CONFIG",
]
