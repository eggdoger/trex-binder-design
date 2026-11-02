"""Fail-closed installation and target validation for T-REX.

This module validates the *execution contract* without launching an LLM or a
GPU worker.  It is intentionally stricter than an import smoke: enabled
families must have the executables and assets that their dispatch paths use,
and registered targets must match both their JSON constraint and PDB content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from Bio.PDB import PDBParser

from .capability_registry import default_registry
from .backends.complexa import inspect_complexa_target
from .provenance import verify_model_manifest
from .resource_paths import embedded_runtime_root, publication_data_path
from .targets import _load_json_object, validate_target_constraint


REPO_ROOT = embedded_runtime_root()
TARGET_REGISTRY = publication_data_path(REPO_ROOT, "config/targets/registry.json")
TARGET_ASSET_MANIFEST = publication_data_path(
    REPO_ROOT, "config/targets/assets.sha256.json"
)
PRODUCTION_STACK = publication_data_path(
    REPO_ROOT, "config/reproducibility/production_stack.json"
)
AF2_ASSET_MANIFEST = publication_data_path(
    REPO_ROOT, "config/reproducibility/af2_parameters_manifest.json"
)
PROTEINMPNN_ASSET_MANIFEST = publication_data_path(
    REPO_ROOT, "config/reproducibility/proteinmpnn_weights_manifest.json"
)
COMPLEXA_ENVIRONMENT = publication_data_path(
    REPO_ROOT, "config/reproducibility/complexa_environment.json"
)
DEFAULT_FAMILIES = (
    "bindcraft",
    "boltzgen",
    "complexa_beam",
    "complexa_best_of_n",
    "complexa_fk_steering",
    "complexa_mcts",
    "proteinmpnn_redesign",
    "structure_refilter",
)
INSTALLATION_VALIDATION_SCHEMA_VERSION = "binder_runtime.installation-validation.v1"

# BindCraft's ColabDesign animation calls Matplotlib's HTML5 writer. Preserve
# its effective rcParams (including configured executable/arguments), while
# using only a headless 16x16 two-frame CPU fixture and no backend/model imports.
_BINDCRAFT_ANIMATION_PROBE = """
import json, sys, tempfile
tempfile.tempdir = sys.argv[1]
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.patches import Rectangle
figure = plt.figure(figsize=(0.25, 0.25), dpi=64)
axes = figure.add_axes([0, 0, 1, 1])
axes.set_axis_off()
frames = []
for color in ('black', 'white'):
    artist = axes.add_patch(Rectangle((0, 0), 1, 1, facecolor=color, animated=True))
    frames.append([artist])
try:
    video = ArtistAnimation(figure, frames, interval=1000).to_html5_video()
    if 'data:video/mp4;base64,' not in video:
        raise RuntimeError('HTML5 animation did not contain an encoded MP4')
    print(json.dumps({'writer': matplotlib.rcParams['animation.writer'],
                     'ffmpeg_path': matplotlib.rcParams['animation.ffmpeg_path'],
                     'html_bytes': len(video)}))
finally:
    plt.close(figure)
"""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.status == "ok" or not self.required


def _env_path(name: str, default: Path | None = None) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if raw:
        return Path(raw).expanduser()
    return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _family_set(value: str | Iterable[str]) -> set[str]:
    items = value.split(",") if isinstance(value, str) else value
    return {str(item).strip() for item in items if str(item).strip()}


def _which(explicit: str | None, fallback: str) -> str | None:
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(fallback)


def _bindcraft_animation_check(
    repo: Path, environment_root: Path, *, required: bool, timeout_s: float = 30
) -> Check:
    """Exercise the H264 encoder used by BindCraft's HTML5 trajectory animation."""
    name = "BindCraft animation encoding"
    settings_path = (
        repo / "settings_advanced" / "default_4stage_multimer_mpnn_hardtarget.json"
    )
    failure = "fail" if required else "warn"
    try:
        settings = json.loads(settings_path.read_text())
        if not isinstance(settings, dict) or not isinstance(
            settings.get("save_design_animations"), bool
        ):
            raise ValueError("save_design_animations must be a boolean")
    except (OSError, UnicodeError, ValueError) as exc:
        return Check(
            name, failure,
            f"cannot read animation setting {settings_path}: {exc}", required,
        )
    # This setting is inherited unchanged by prepare_bindcraft_launch: it is
    # deliberately absent from ADVANCED_OVERRIDE_KEYS.
    if not settings["save_design_animations"]:
        return Check(
            name, "ok", f"disabled in {settings_path}; no encoder required", required,
        )

    from .backends.process_management import isolated_subprocess_environment

    try:
        environment = isolated_subprocess_environment(environment_root)
        # An installation check must not acquire a CUDA context for encoding.
        environment["CUDA_VISIBLE_DEVICES"] = ""
        executable = environment_root / "bin" / "python"
        # The outer directory also contains Matplotlib's nested video tempdir,
        # so timeout cleanup works even when its Python finally cannot run.
        with tempfile.TemporaryDirectory(prefix="trex-bindcraft-video-") as temporary:
            with (Path(temporary) / "stdout").open("w+b") as stdout, (
                Path(temporary) / "stderr"
            ).open("w+b") as stderr:
                with subprocess.Popen(
                    [str(executable), "-c", _BINDCRAFT_ANIMATION_PROBE, temporary],
                    env=environment, stdout=stdout, stderr=stderr,
                    start_new_session=True,
                ) as process:
                    try:
                        process.wait(timeout=timeout_s)
                    except subprocess.TimeoutExpired:
                        # start_new_session guarantees this group cannot be the
                        # controller's. Kill both Python and a stalled encoder.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=5)
                        raise
                stdout.seek(0)
                observed = stdout.read(65536).decode(errors="replace").strip()
                stderr.seek(0, os.SEEK_END)
                stderr.seek(max(0, stderr.tell() - 2000))
                diagnostic = stderr.read().decode(errors="replace").strip()
            if process.returncode:
                return Check(
                    name, failure,
                    f"{executable}: HTML5 H264 encode exited {process.returncode}; {diagnostic}",
                    required,
                )
            report = json.loads(observed)
            if not isinstance(report, dict) or not report.get("html_bytes"):
                raise ValueError("animation probe produced no encoded HTML5 video")
    except subprocess.TimeoutExpired:
        return Check(
            name, failure, f"H264 encoding timed out after {timeout_s}s", required,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return Check(name, failure, f"H264 encoding failed: {exc}", required)
    return Check(
        name, "ok",
        f"{executable}: encoded two HTML5 H264 frames; writer={report.get('writer')} "
        f"ffmpeg_path={report.get('ffmpeg_path')} html_bytes={report['html_bytes']}",
        required,
    )


def _git_head(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_diff_sha256(path: Path) -> str | None:
    """Hash the tracked working-tree diff using the provenance convention."""
    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
            cwd=path,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    diff = proc.stdout.strip()
    return hashlib.sha256(diff.encode()).hexdigest() if diff else None


def _tracked_source_check(path: Path, component: str) -> Check:
    """Allow only Complexa's recorded dependency metadata installation patch."""
    name = f"{component} tracked source"
    if component != "proteina_complexa":
        try:
            unchanged = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--"], cwd=path,
                capture_output=True, timeout=15,
            ).returncode == 0
        except (OSError, subprocess.SubprocessError):
            unchanged = False
        return Check(name, "ok" if unchanged else "fail",
                     "unchanged" if unchanged else "changed or unavailable")

    try:
        # Inspect both final working-tree contents and the index. Force mode
        # checks even in repositories configured with core.filemode=false.
        changes = []
        for cached in (False, True):
            result = subprocess.run(
                ["git", "-c", "core.filemode=true", "diff", "--raw", "-z",
                 "--no-ext-diff", "--no-textconv", "--no-renames",
                 "--ignore-submodules=none", *(["--cached"] if cached else []),
                 "HEAD", "--"], cwd=path, capture_output=True, timeout=15,
            )
            if result.returncode != 0:
                raise ValueError("tracked diff unavailable")
            changes.append(result.stdout)
        if not any(changes):
            return Check(name, "ok", "unchanged")
        for changed in changes:
            if not changed:
                continue
            entries = changed.rstrip(b"\0").split(b"\0")
            if (len(entries) != 2 or entries[1] != b"pyproject.toml"
                    or not re.fullmatch(
                        rb":100644 100644 [0-9a-f]+ [0-9a-f]+ M", entries[0])):
                raise ValueError("tracked changes extend beyond dependency metadata contents")
        profile = json.loads(COMPLEXA_ENVIRONMENT.read_text())
        expected = profile.get("patched_pyproject_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("installation profile lacks a valid patched_pyproject_sha256")
        project = path / "pyproject.toml"
        if project.is_symlink() or not project.is_file() or _sha256(project) != expected:
            raise ValueError("pyproject.toml differs from the recorded installation patch")
        if changes[1]:
            staged = subprocess.run(
                ["git", "show", ":pyproject.toml"], cwd=path,
                capture_output=True, timeout=15,
            )
            if staged.returncode != 0 or hashlib.sha256(staged.stdout).hexdigest() != expected:
                raise ValueError("staged pyproject.toml differs from the recorded installation patch")
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as exc:
        return Check(name, "fail", f"changed or unavailable: {exc}")
    return Check(name, "ok", f"only recorded dependency metadata patch; pyproject_sha256={expected}")


def _equivalent_source_checks(
    path: Path, component: str, source_trees: dict[str, str]
) -> list[Check]:
    """Verify unchanged source subtrees in an explicitly permitted public checkout."""
    checks: list[Check] = []
    if not source_trees:
        return [
            Check(f"{component} source equivalence", "fail", "no source trees declared")
        ]
    for relative, expected in sorted(source_trees.items()):
        try:
            tree = subprocess.run(
                ["git", "rev-parse", "--verify", f"HEAD:{relative}"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            diff = subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", relative],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            observed = tree.stdout.strip() if tree.returncode == 0 else None
            ok = observed == expected and diff.returncode == 0
            detail = (
                f"tree={observed or '<unavailable>'} expected={expected} "
                f"tracked_source_unchanged={diff.returncode == 0}"
            )
        except (OSError, subprocess.SubprocessError) as exc:
            ok, detail = False, str(exc)
        checks.append(
            Check(f"{component} source {relative}", "ok" if ok else "fail", detail)
        )
    return checks


def _actual_source_tree_check(label: str, source_dir: Path, expected: str | None) -> Check:
    """Verify the exact directory Python will import, irrespective of repo layout."""
    try:
        if not source_dir.is_dir() or not expected:
            raise ValueError(f"missing source directory or reference tree: {source_dir}")
        root_result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=source_dir, capture_output=True, text=True, timeout=15)
        if root_result.returncode:
            raise ValueError("external backend source must have verifiable Git metadata")
        git_root = Path(root_result.stdout.strip()).resolve()
        relative = source_dir.resolve().relative_to(git_root).as_posix()
        tree_ref = "HEAD^{tree}" if relative == "." else f"HEAD:{relative}"
        tree = subprocess.run(["git", "rev-parse", "--verify", tree_ref], cwd=git_root, capture_output=True, text=True, timeout=15)
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=git_root, capture_output=True, timeout=15)
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", relative], cwd=git_root, capture_output=True, text=True, timeout=15)
        untracked_code = [name for name in untracked.stdout.splitlines() if Path(name).suffix in {".py", ".so", ".pyd"}]
        observed = tree.stdout.strip() if tree.returncode == 0 else None
        ok = observed == expected and diff.returncode == 0 and untracked.returncode == 0 and not untracked_code
        detail = f"path={source_dir} tree={observed or '<unavailable>'} expected={expected} tracked_unchanged={diff.returncode == 0} untracked_code={untracked_code}"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        ok, detail = False, str(exc)
    return Check(f"{label} source tree", "ok" if ok else "fail", detail)


def _target_checks(
    target: str,
    *,
    target_config: Path | None,
    target_pdb: Path | None,
    asset_root: Path | None,
    verify_asset_hash: bool,
) -> tuple[list[Check], Path | None, Path | None, str | None]:
    checks: list[Check] = []
    if TARGET_REGISTRY.is_file():
        registry = json.loads(TARGET_REGISTRY.read_text())
        entry = registry.get("targets", {}).get(target)
    else:
        registry = {}
        entry = None
    if entry is None and target_config is None:
        detail = (
            f"unknown target without --target-config: {target}"
            if TARGET_REGISTRY.is_file()
            else "bundled target registry is unavailable; pass --target-config and --target-pdb"
        )
        return [Check("target registry", "fail", detail)], None, None, None

    if entry is None:
        assert target_config is not None
        config = target_config
        pdb = target_pdb
        expected_id = None
        checks.append(
            Check(
                "target registry",
                "warn",
                "custom target; registry hash is unavailable",
                required=False,
            )
        )
    else:
        config = target_config or publication_data_path(
            REPO_ROOT, Path("config/targets") / entry["config"]
        )
        if target_pdb is not None:
            pdb = target_pdb
        elif asset_root is not None:
            direct = asset_root / entry["pdb"]
            nested = asset_root / "targets" / entry["pdb"]
            pdb = direct if direct.is_file() or not nested.is_file() else nested
        else:
            pdb = None
        expected_id = str(entry["target_id"])
    checks.append(
        Check("target config", "ok" if config.is_file() else "fail", str(config))
    )
    if not config.is_file():
        return checks, config, pdb, expected_id

    try:
        payload = _load_json_object(config, label="target constraint")
        validate_target_constraint(payload)
    except ValueError as exc:
        checks.append(Check("target constraint schema", "fail", str(exc)))
        return checks, config, pdb, expected_id
    actual_id = str(payload.get("target_id", ""))
    if expected_id is None:
        expected_id = actual_id or None
    checks.append(
        Check(
            "target id",
            "ok" if actual_id == expected_id else "fail",
            f"expected={expected_id} observed={actual_id or '<missing>'}",
        )
    )
    chains = [str(chain) for chain in payload.get("chain_ids", [])]
    checks.append(
        Check(
            "target chains",
            "ok" if chains and len(chains) == len(set(chains)) else "fail",
            ",".join(chains) if chains else "chain_ids is empty",
        )
    )
    if pdb is None:
        checks.append(
            Check(
                "target PDB",
                "warn",
                "not configured; set TREX_TARGET_ASSET_ROOT or pass --target-pdb",
                required=False,
            )
        )
        return checks, config, pdb, expected_id
    checks.append(Check("target PDB", "ok" if pdb.is_file() else "fail", str(pdb)))
    if not pdb.is_file():
        return checks, config, pdb, expected_id

    try:
        structure = PDBParser(QUIET=True).get_structure(target, str(pdb))
        model = next(structure.get_models())
        residues_by_chain = {
            chain.id: {
                (int(residue.id[1]), str(residue.id[2]).strip())
                for residue in chain.get_residues()
                if residue.id[0] == " "
            }
            for chain in model
        }
    except Exception as exc:
        checks.append(Check("PDB parse", "fail", f"{type(exc).__name__}: {exc}"))
        return checks, config, pdb, expected_id

    missing_chains = [chain for chain in chains if chain not in residues_by_chain]
    checks.append(
        Check(
            "configured chains in PDB",
            "fail" if missing_chains else "ok",
            f"missing={missing_chains}" if missing_chains else ",".join(chains),
        )
    )

    bad_hotspots: list[str] = []
    for hotspot in payload.get("hotspots", []):
        text = str(hotspot)
        match = re.fullmatch(r"(.)(-?\d+)([A-Za-z]?)", text)
        if match is None:
            bad_hotspots.append(text)
            continue
        chain, residue_no, insertion = (
            match.group(1),
            int(match.group(2)),
            match.group(3),
        )
        if (residue_no, insertion) not in residues_by_chain.get(chain, set()):
            bad_hotspots.append(text)
    checks.append(
        Check(
            "hotspots in PDB",
            "fail" if bad_hotspots else "ok",
            f"missing_or_invalid={bad_hotspots}"
            if bad_hotspots
            else ",".join(payload.get("hotspots", [])),
        )
    )

    if verify_asset_hash and entry is not None and TARGET_ASSET_MANIFEST.is_file():
        manifest = json.loads(TARGET_ASSET_MANIFEST.read_text())
        expected_sha = manifest.get("targets", {}).get(target, {}).get("sha256")
        observed_sha = _sha256(pdb)
        checks.append(
            Check(
                "target PDB SHA256",
                "ok" if expected_sha and observed_sha == expected_sha else "fail",
                f"expected={expected_sha or '<missing>'} observed={observed_sha}",
            )
        )
    return checks, config, pdb, expected_id


def validate_install(
    *,
    target: str,
    enabled_families: str | Iterable[str] = DEFAULT_FAMILIES,
    target_config: Path | None = None,
    target_pdb: Path | None = None,
    asset_root: Path | None = None,
    require_backends: bool = False,
    require_model: bool = False,
    verify_asset_hash: bool = True,
    verify_backend_revisions: bool = False,
    verify_checkpoint_content: bool = False,
) -> list[Check]:
    """Return all preflight checks; callers decide how to render failures."""
    families = _family_set(enabled_families)
    checks, resolved_target_config, resolved_target_pdb, resolved_target_id = _target_checks(
        target,
        target_config=target_config,
        target_pdb=target_pdb,
        asset_root=asset_root,
        verify_asset_hash=verify_asset_hash,
    )

    try:
        registry = default_registry()
    except Exception as exc:  # noqa: BLE001 - malformed operator plugin.
        checks.append(
            Check(
                "backend extension registry",
                "fail",
                f"{type(exc).__name__}: {exc}",
            )
        )
        registry = None
    if registry is not None:
        unknown = sorted(family for family in families if registry.get(family) is None)
        unavailable = sorted(
            family
            for family in families
            if (cap := registry.get(family)) is not None
            and cap.availability != "available"
        )
        checks.append(
            Check(
                "enabled family registry",
                "fail" if unknown or unavailable else "ok",
                f"unknown={unknown} unavailable={unavailable}"
                if unknown or unavailable
                else ",".join(sorted(families)) or "none",
            )
        )
        diagnostic = sorted(
            family
            for family in families
            if (cap := registry.get(family)) is not None and cap.outputs_diagnostic_only
        )
        if diagnostic and "structure_refilter" not in families:
            # Match the controller's fail-safe auto-enable behavior, and make
            # the resulting AF2 dependency visible during preflight.
            families.add("structure_refilter")
            checks.append(
                Check(
                    "canonical score-conversion route",
                    "warn",
                    f"auto-required for diagnostic families: {','.join(diagnostic)}",
                    required=False,
                )
            )
        for family in sorted(families):
            cap = registry.get(family)
            if cap is None or cap.preflight is None:
                continue
            try:
                ok, detail = cap.preflight()
            except Exception as exc:  # noqa: BLE001 - operator hook boundary.
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            checks.append(
                Check(
                    f"{family} adapter preflight",
                    "ok" if ok else ("fail" if require_backends else "warn"),
                    detail,
                    required=require_backends,
                )
            )

    foldseek = _which(os.environ.get("TREX_FOLDSEEK_BIN"), "foldseek")
    checks.append(
        Check(
            "Foldseek executable",
            "ok" if foldseek else ("fail" if require_backends else "warn"),
            foldseek or "not found; set TREX_FOLDSEEK_BIN or PATH",
            required=require_backends,
        )
    )
    mmseqs = _which(os.environ.get("TREX_MMSEQS_BIN"), "mmseqs")
    checks.append(
        Check(
            "MMseqs2 executable",
            "ok" if mmseqs else ("fail" if require_backends else "warn"),
            mmseqs or "not found; set TREX_MMSEQS_BIN or PATH",
            required=require_backends,
        )
    )
    if verify_backend_revisions and PRODUCTION_STACK.is_file():
        tool_stack = json.loads(PRODUCTION_STACK.read_text()).get("components", {})
        for label, executable, component in (
            ("Foldseek version", foldseek, "foldseek"),
            ("MMseqs2 version", mmseqs, "mmseqs2"),
        ):
            if not executable:
                continue
            expected = str(tool_stack.get(component, {}).get("version", ""))
            try:
                observed = subprocess.run(
                    [executable, "version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                detail = (observed.stdout or observed.stderr).strip()
                ok = observed.returncode == 0 and bool(expected) and expected in detail
            except (OSError, subprocess.SubprocessError) as exc:
                ok, detail = False, str(exc)
            checks.append(
                Check(
                    label,
                    "ok" if ok else "fail",
                    f"expected={expected or '<missing>'} observed={detail or '<unavailable>'}",
                )
            )

    need_complexa = any(family.startswith("complexa_") for family in families)
    need_af2 = bool({"structure_refilter", "proteinmpnn_redesign"} & families)
    need_bindcraft = "bindcraft" in families
    need_boltzgen = "boltzgen" in families
    # Imported lazily to keep config -> validation -> runtime free of a cycle.
    from .campaign.runtime.paths import RuntimePaths
    runtime_paths = RuntimePaths.from_environment()
    complexa = runtime_paths.complexa_repo
    complexa_python = runtime_paths.complexa_python
    bindcraft = runtime_paths.bindcraft_repo
    bindcraft_env = runtime_paths.bindcraft_env
    boltzgen = runtime_paths.boltzgen_repo
    boltzgen_bin = runtime_paths.boltzgen_binary
    af2_python = runtime_paths.af2_python or complexa_python
    mpnn_python = runtime_paths.proteinmpnn_python or complexa_python

    def path_check(
        name: str, path: Path | None, required: bool, kind: str = "exists"
    ) -> None:
        if kind == "file":
            ok = bool(path and path.is_file())
        elif kind == "exec":
            ok = bool(path and path.is_file() and os.access(path, os.X_OK))
        else:
            ok = bool(path and path.exists())
        checks.append(
            Check(
                name,
                "ok" if ok else ("fail" if required and require_backends else "warn"),
                str(path) if path else "not configured",
                required=required and require_backends,
            )
        )

    path_check("Proteina-Complexa checkout", complexa, need_complexa)
    path_check(
        "Proteina-Complexa env.sh",
        complexa / "env.sh" if complexa else None,
        need_complexa,
        "file",
    )
    path_check(
        "Complexa Python", complexa_python, need_complexa, "exec"
    )
    path_check(
        "Complexa checkpoint",
        complexa / "ckpts" / "complexa.ckpt" if complexa else None,
        need_complexa,
        "file",
    )
    path_check(
        "Complexa AE checkpoint",
        complexa / "ckpts" / "complexa_ae.ckpt" if complexa else None,
        need_complexa,
        "file",
    )
    if need_complexa:
        target_inputs_ready = (
            complexa and (complexa / "configs/targets/targets_dict.yaml").is_file()
            and resolved_target_config and resolved_target_config.is_file()
            and resolved_target_pdb and resolved_target_pdb.is_file()
        )
        if target_inputs_ready:
            try:
                payload = _load_json_object(resolved_target_config, label="target constraint")
                detail = inspect_complexa_target(
                    repo=complexa, target_id=resolved_target_id or str(payload["target_id"]),
                    target_pdb=resolved_target_pdb,
                    chain_ids=tuple(payload.get("chain_ids", [])),
                    hotspots=tuple(payload.get("hotspots", [])),
                )
                checks.append(Check("Complexa target binding", "ok", json.dumps(detail, sort_keys=True)))
            except (OSError, ValueError, KeyError) as exc:
                checks.append(Check("Complexa target binding", "fail", str(exc)))
        else:
            checks.append(Check(
                "Complexa target binding", "fail" if require_backends else "warn",
                "requires backend targets_dict.yaml and readable campaign target PDB/constraint",
                required=require_backends,
            ))
    path_check("AF2 Python", af2_python, need_af2, "exec")
    path_check("ColabDesign source", runtime_paths.community_models_root / "colabdesign" / "__init__.py", need_af2, "file")
    path_check("ProteinMPNN Python", mpnn_python, "proteinmpnn_redesign" in families, "exec")
    path_check("ProteinMPNN entrypoint", runtime_paths.proteinmpnn_dir / "protein_mpnn_run.py", "proteinmpnn_redesign" in families, "file")
    af2_root = runtime_paths.af2_data_dir
    path_check("AF2 parameters", af2_root, need_af2)
    for model_index in range(1, 6):
        path_check(
            f"AF2 multimer-v3 model {model_index}",
            af2_root / f"params_model_{model_index}_multimer_v3.npz"
            if af2_root
            else None,
            need_af2,
            "file",
        )
    mpnn_root = runtime_paths.proteinmpnn_weights_dir
    for model_name in ("v_48_002", "v_48_010", "v_48_020", "v_48_030"):
        path_check(
            f"ProteinMPNN weight {model_name}",
            mpnn_root / f"{model_name}.pt" if mpnn_root else None,
            "proteinmpnn_redesign" in families,
            "file",
        )

    def manifest_check(name: str, root: Path | None, manifest_path: Path) -> None:
        path_check(f"{name} manifest", manifest_path, True, "file")
        if not root or not root.is_dir() or not manifest_path.is_file():
            return
        try:
            digest = verify_model_manifest(
                root,
                json.loads(manifest_path.read_text()),
                verify_content=verify_checkpoint_content,
            )
        except Exception as exc:
            checks.append(
                Check(
                    f"{name} verification",
                    "fail" if require_backends else "warn",
                    str(exc),
                    required=require_backends,
                )
            )
        else:
            mode = "full SHA256" if verify_checkpoint_content else "manifest + sizes"
            checks.append(
                Check(
                    f"{name} verification",
                    "ok",
                    f"{digest} ({mode})",
                    required=require_backends,
                )
            )

    if need_af2:
        manifest_check("AF2 parameters", af2_root, AF2_ASSET_MANIFEST)
    if "proteinmpnn_redesign" in families:
        manifest_check("ProteinMPNN weights", mpnn_root, PROTEINMPNN_ASSET_MANIFEST)

    path_check(
        "BindCraft checkout",
        bindcraft / "bindcraft.py" if bindcraft else None,
        need_bindcraft,
        "file",
    )
    path_check(
        "BindCraft Python",
        bindcraft_env / "bin" / "python" if bindcraft_env else None,
        need_bindcraft,
        "exec",
    )
    if need_bindcraft:
        checks.append(_bindcraft_animation_check(
            bindcraft, bindcraft_env, required=require_backends,
        ))
    path_check("BoltzGen checkout", boltzgen, need_boltzgen)
    path_check("BoltzGen executable", boltzgen_bin, need_boltzgen, "exec")

    if require_model:
        checks.append(Check("decision provider", "fail", "Local model serving is unsupported; use the external Codex mailbox provider."))

    if verify_backend_revisions:
        if not PRODUCTION_STACK.is_file():
            checks.append(
                Check(
                    "production stack manifest",
                    "fail",
                    f"not found: {PRODUCTION_STACK}",
                )
            )
            return checks
        stack = json.loads(PRODUCTION_STACK.read_text())
        paths = {
            "proteina_complexa": (complexa, need_complexa),
            "bindcraft": (bindcraft, need_bindcraft),
            "boltzgen": (boltzgen, need_boltzgen),
        }
        community = stack.get("components", {}).get("proteina_complexa_community", {})
        source_trees = community.get("equivalent_source_trees", {})
        for label, source_dir, subtree, enabled in (
            ("ColabDesign", runtime_paths.community_models_root / "colabdesign", "community_models/colabdesign", need_af2),
            ("ProteinMPNN", runtime_paths.proteinmpnn_dir, "community_models/ProteinMPNN", "proteinmpnn_redesign" in families),
        ):
            if enabled:
                checks.append(_actual_source_tree_check(label, source_dir, source_trees.get(subtree)))
        for name, (path, enabled) in paths.items():
            if not enabled:
                continue
            component = stack.get("components", {}).get(name, {})
            expected = component.get("git_commit")
            if path and path.is_dir() and expected:
                observed = _git_head(path)
                exact_revision = observed == expected
                equivalent_checks: list[Check] = []
                if not exact_revision and observed in component.get(
                    "equivalent_git_commits", []
                ):
                    equivalent_checks = _equivalent_source_checks(
                        path, name, component.get("equivalent_source_trees", {})
                    )
                equivalent_revision = bool(equivalent_checks) and all(
                    check.passed for check in equivalent_checks
                )
                checks.append(
                    Check(
                        f"{name} revision",
                        "ok" if exact_revision or equivalent_revision else "fail",
                        f"expected={expected} observed={observed or '<unavailable>'}"
                        + (
                            "; identical declared source trees"
                            if equivalent_revision
                            else ""
                        ),
                    )
                )
                checks.extend(equivalent_checks)
            else:
                checks.append(Check(f"{name} revision", "fail", f"missing backend directory or reference revision: {path}"))
            if name != "bindcraft" and path and path.is_dir():
                checks.append(_tracked_source_check(path, name))
            if name == "bindcraft" and path and path.is_dir():
                component = stack.get("components", {}).get(name, {})
                patch_name = component.get("patch")
                expected_patch_sha = component.get("patch_file_sha256")
                if patch_name and expected_patch_sha:
                    patch_path = publication_data_path(REPO_ROOT, str(patch_name))
                    observed_patch_sha = (
                        _sha256(patch_path) if patch_path.is_file() else None
                    )
                    checks.append(
                        Check(
                            "bindcraft patch artifact",
                            "ok"
                            if observed_patch_sha == expected_patch_sha
                            else "fail",
                            f"expected={expected_patch_sha} "
                            f"observed={observed_patch_sha or '<missing>'}",
                        )
                    )
                expected_diff = component.get("production_tracked_diff_sha256")
                if expected_diff:
                    observed_diff = _git_diff_sha256(path)
                    checks.append(
                        Check(
                            "bindcraft tracked patch",
                            "ok" if observed_diff == expected_diff else "fail",
                            f"expected={expected_diff} "
                            f"observed={observed_diff or '<clean-or-unavailable>'}",
                        )
                    )
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", required=True, help="registered key or custom target label"
    )
    parser.add_argument("--target-config", type=Path)
    parser.add_argument("--target-pdb", type=Path)
    parser.add_argument(
        "--asset-root", type=Path, default=_env_path("TREX_TARGET_ASSET_ROOT")
    )
    parser.add_argument(
        "--enabled-families",
        default=os.environ.get("TREX_ENABLED_FAMILIES", ",".join(DEFAULT_FAMILIES)),
    )
    parser.add_argument("--require-backends", action="store_true")
    parser.add_argument("--skip-asset-hash", action="store_true")
    parser.add_argument("--verify-backend-revisions", action="store_true")
    parser.add_argument(
        "--verify-checkpoint-content",
        action="store_true",
        help="rehash model/checkpoint bytes; slower than the default file/size check",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checks = validate_install(
        target=args.target,
        enabled_families=args.enabled_families,
        target_config=args.target_config,
        target_pdb=args.target_pdb,
        asset_root=args.asset_root,
        require_backends=args.require_backends,
        verify_asset_hash=not args.skip_asset_hash,
        verify_backend_revisions=args.verify_backend_revisions,
        verify_checkpoint_content=args.verify_checkpoint_content,
    )
    failed = [check for check in checks if check.required and check.status == "fail"]
    if args.as_json:
        print(
            json.dumps(
                {
                    "ok": not failed,
                    "schema_version": INSTALLATION_VALIDATION_SCHEMA_VERSION,
                    "failure_count": len(failed),
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
            )
        )
    else:
        width = max(len(check.name) for check in checks)
        for check in checks:
            print(f"{check.status.upper():4}  {check.name:<{width}}  {check.detail}")
        print(
            "\nPreflight passed."
            if not failed
            else f"\nPreflight failed: {len(failed)} required check(s).",
            file=sys.stdout if not failed else sys.stderr,
        )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
