"""Immutable backend locations, separate from the skill's embedded Python code."""
from __future__ import annotations
import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from ...resource_paths import embedded_runtime_root

BACKEND_PATH_ENVIRONMENT_VARIABLES: Mapping[str, str] = MappingProxyType({
    "external_root": "TREX_EXTERNAL_ROOT",
    "complexa_repo": "TREX_COMPLEXA_REPO",
    "legacy_complexa_repo": "TREX_LEGACY_COMPLEXA_REPO",
    "complexa_python": "TREX_COMPLEXA_PYTHON",
    "community_models_path": "TREX_COMMUNITY_ROOT",
    "af2_data_path": "TREX_AF2_DATA_DIR",
    "af2_python": "TREX_AF2_PYTHON",
    "proteinmpnn_path": "TREX_PROTEINMPNN_DIR",
    "proteinmpnn_weights_path": "TREX_PROTEINMPNN_WEIGHTS_DIR",
    "proteinmpnn_python": "TREX_PROTEINMPNN_PYTHON",
    "bindcraft_repo": "TREX_BINDCRAFT_REPO",
    "bindcraft_env": "TREX_BINDCRAFT_ENV",
    "boltzgen_repo": "TREX_BOLTZGEN_REPO",
    "boltzgen_binary": "TREX_BOLTZGEN_BIN",
    "boltzgen_cache": "TREX_BOLTZGEN_CACHE",
    "foldseek_binary": "TREX_FOLDSEEK_BIN",
    "mmseqs_binary": "TREX_MMSEQS_BIN",
})
EXECUTABLE_BACKEND_PATH_FIELDS = frozenset({
    "complexa_python", "af2_python", "proteinmpnn_python", "boltzgen_binary",
    "foldseek_binary", "mmseqs_binary",
})


def resolve_environment_path(environment_name: str, default: str | Path, *,
                             environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    return Path(values.get(environment_name, "").strip() or default).expanduser().resolve()


def resolve_environment_executable_path(environment_name: str, default: str | Path, *,
                                        environment: Mapping[str, str] | None = None) -> Path:
    """Keep virtualenv executable symlinks, whose location selects the environment."""
    values = os.environ if environment is None else environment
    return Path(os.path.abspath(Path(values.get(environment_name, "").strip() or default).expanduser()))


@dataclass(frozen=True)
class RuntimePaths:
    # Compatibility name: this is always skill/scripts, never an upstream checkout.
    repo_root: Path
    external_root: Path
    complexa_repo: Path
    legacy_complexa_repo: Path
    complexa_python: Path
    bindcraft_repo: Path
    bindcraft_env: Path
    boltzgen_repo: Path
    boltzgen_binary: Path
    boltzgen_cache: Path
    foldseek_binary: Path | None = None
    mmseqs_binary: Path | None = None
    community_models_path: Path | None = None
    af2_data_path: Path | None = None
    af2_python: Path | None = None
    proteinmpnn_path: Path | None = None
    proteinmpnn_weights_path: Path | None = None
    proteinmpnn_python: Path | None = None

    @property
    def community_models_root(self) -> Path:
        return self.community_models_path or self.legacy_complexa_repo / "community_models"

    @property
    def af2_data_dir(self) -> Path:
        return self.af2_data_path or self.community_models_root / "ckpts" / "AF2"

    @property
    def proteinmpnn_dir(self) -> Path:
        return self.proteinmpnn_path or self.community_models_root / "ProteinMPNN"

    @property
    def proteinmpnn_weights_dir(self) -> Path:
        return self.proteinmpnn_weights_path or self.proteinmpnn_dir / "vanilla_model_weights"

    @property
    def proteinmpnn_weights(self) -> Path:
        return self.proteinmpnn_weights_dir / "v_48_020.pt"

    @property
    def foldseek_command(self) -> str:
        return str(self.foldseek_binary) if self.foldseek_binary else "foldseek"

    @property
    def mmseqs_command(self) -> str:
        return str(self.mmseqs_binary) if self.mmseqs_binary else "mmseqs"

    def as_dict(self) -> dict[str, str | None]:
        output = {item.name: str(getattr(self, item.name)) if getattr(self, item.name) is not None else None
                  for item in fields(self)}
        for name in ("community_models_root", "af2_data_dir", "proteinmpnn_dir",
                     "proteinmpnn_weights_dir", "proteinmpnn_weights"):
            output[name] = str(getattr(self, name))
        return output

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None, *,
                         default_repo_root: str | Path | None = None) -> "RuntimePaths":
        """Resolve external resources; legacy source-root settings cannot redirect code."""
        values = os.environ if environment is None else environment

        def path(name: str, default: Path | None) -> Path | None:
            env_name = BACKEND_PATH_ENVIRONMENT_VARIABLES[name]
            if not values.get(env_name, "").strip() and default is None:
                return None
            resolver = (resolve_environment_executable_path if name in EXECUTABLE_BACKEND_PATH_FIELDS
                        else resolve_environment_path)
            return resolver(env_name, default or "", environment=values)

        external = path("external_root", Path.home() / ".cache" / "trex-binder-design" / "external")
        assert external is not None
        complexa = path("complexa_repo", external / "Proteina-Complexa")
        bindcraft = path("bindcraft_repo", external / "BindCraft")
        boltzgen = path("boltzgen_repo", external / "BoltzGen")
        assert complexa is not None and bindcraft is not None and boltzgen is not None
        python = path("complexa_python", complexa / ".venv" / "bin" / "python")
        search_path = values.get("PATH", "")
        foldseek = shutil.which("foldseek", path=search_path)
        mmseqs = shutil.which("mmseqs", path=search_path)
        return cls(
            repo_root=embedded_runtime_root(), external_root=external,
            complexa_repo=complexa, legacy_complexa_repo=path("legacy_complexa_repo", complexa),
            complexa_python=python, bindcraft_repo=bindcraft,
            bindcraft_env=path("bindcraft_env", bindcraft / ".venv"), boltzgen_repo=boltzgen,
            boltzgen_binary=path("boltzgen_binary", boltzgen / ".venv" / "bin" / "boltzgen"),
            boltzgen_cache=path("boltzgen_cache", external / "checkpoints" / "boltzgen"),
            foldseek_binary=path("foldseek_binary", Path(foldseek) if foldseek else None),
            mmseqs_binary=path("mmseqs_binary", Path(mmseqs) if mmseqs else None),
            community_models_path=path("community_models_path", None),
            af2_data_path=path("af2_data_path", None), af2_python=path("af2_python", python),
            proteinmpnn_path=path("proteinmpnn_path", None),
            proteinmpnn_weights_path=path("proteinmpnn_weights_path", None),
            proteinmpnn_python=path("proteinmpnn_python", python),
        )
