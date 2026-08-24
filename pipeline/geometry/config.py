"""Loader for the versioned geometry configuration.

Every consequential geometry threshold lives in `config/geometry_config_v0.1.json`
with its rationale and source category. Code reads values through this loader so
that no threshold can be quietly hardcoded at a call site, and so the exact
configuration behind any output can be identified by hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "geometry_config_v0.1.json"


class ConfigError(Exception):
    """The geometry configuration is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class GeometryConfig:
    config_id: str
    path: Path
    raw: dict
    sha256: str

    @property
    def frozen(self) -> bool:
        return bool(self.raw.get("frozen"))

    def get(self, name: str):
        """Return a parameter's value, refusing anything undocumented."""
        parameters = self.raw["parameters"]
        if name not in parameters:
            raise ConfigError(
                f"'{name}' is not defined in {self.config_id}. Add it to "
                f"{self.path.name} with a value, units, rationale, and source "
                f"category rather than hardcoding it.")
        entry = parameters[name]
        for field in ("value", "units", "rationale", "sourceCategory"):
            if field not in entry:
                raise ConfigError(
                    f"parameter '{name}' in {self.config_id} is missing "
                    f"required field '{field}'")
        return entry["value"]

    def provenance(self, *names: str) -> dict:
        """A compact record of exactly which values produced an output."""
        return {
            "geometryConfigId": self.config_id,
            "geometryConfigHash": self.sha256,
            "parameters": {
                name: {
                    "value": self.raw["parameters"][name]["value"],
                    "units": self.raw["parameters"][name]["units"],
                    "sourceCategory": self.raw["parameters"][name]["sourceCategory"],
                }
                for name in names
            },
        }


def load_geometry_config(path: Path | None = None) -> GeometryConfig:
    path = Path(path or DEFAULT_CONFIG)
    if not path.is_file():
        raise ConfigError(f"geometry configuration '{path}' does not exist")
    text = path.read_text()
    raw = json.loads(text)
    for field in ("configId", "parameters", "calibrated", "tuningPolicy"):
        if field not in raw:
            raise ConfigError(f"'{path.name}' is missing required field '{field}'")
    if raw.get("calibrated") is not False:
        raise ConfigError(
            "geometry configuration must declare calibrated=false; this POC has no "
            "calibrated parameters and must not imply otherwise")
    return GeometryConfig(
        config_id=raw["configId"],
        path=path,
        raw=raw,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
