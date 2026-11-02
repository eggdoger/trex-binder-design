"""User-facing campaign configuration and inspection APIs.

The scientific controller remains in :mod:`binder_runtime.controller`.  This package
owns the boundary between a user-supplied campaign file and that controller:
loading, validation, target resolution, preflight checks, and concise archive
status output.
"""

from .config import ConfigError, load_campaign
from .models import (
    BackendPaths,
    CampaignConfig,
    MemoryConfig,
    ResolvedCampaign,
)
from .runtime import RuntimePaths

__all__ = [
    "BackendPaths",
    "CampaignConfig",
    "ConfigError",
    "MemoryConfig",
    "ResolvedCampaign",
    "RuntimePaths",
    "load_campaign",
]
