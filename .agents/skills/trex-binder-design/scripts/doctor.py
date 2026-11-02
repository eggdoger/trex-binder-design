#!/usr/bin/env python3
"""Read-only host and standalone skill inventory; never establishes GPU readiness.

Run with --host-only or --skill-dir. Paths refer to this host. This helper never
opens environment profiles, imports backend/runtime code, connects to another
host, installs software, or executes a design job.
"""

import argparse
import datetime
import json
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading


EXPECTED_COMMIT = "bbecc1356556293f8a24ada8dc4483def59379e0"
PROBE_TIMEOUT_SECONDS = 5
MAX_PROBE_OUTPUT_BYTES = 4096
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
SCHEMA_VERSION = "2.0"


def normalized_path(value):
    """Normalize host-local paths without expanding environment variables."""
    return Path(os.path.abspath(value))


def probe(argv):
    """Run an explicit read-only argv with bounded time and captured bytes."""
    captured = bytearray()
    truncated = [False]
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, shell=False,
        )
    except OSError as exc:
        return {"status": "error", "error": type(exc).__name__}

    def drain():
        try:
            while True:
                chunk = process.stdout.read(1024)
                if not chunk:
                    break
                remaining = MAX_PROBE_OUTPUT_BYTES - len(captured)
                captured.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[0] = True
        except (OSError, ValueError):
            pass
        finally:
            process.stdout.close()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait(timeout=1)
    reader.join(timeout=1)
    return {
        "status": "timeout" if timed_out else ("ok" if process.returncode == 0 else "error"),
        "returncode": process.returncode,
        "output": bytes(captured).decode("utf-8", errors="replace").strip(),
        "output_truncated": truncated[0] or reader.is_alive(),
    }


def path_inventory(path):
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "is_file": path.is_file(),
    }


def disk_inventory(path):
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    try:
        usage = shutil.disk_usage(ancestor)
        return {"measured_at": str(ancestor), "total_bytes": usage.total,
                "used_bytes": usage.used, "free_bytes": usage.free}
    except OSError as exc:
        return {"measured_at": str(ancestor), "error": type(exc).__name__}


def inspect_bundle(skill_dir, report, note):
    """Check declared files without executing code or needing a Git checkout."""
    try:
        manifest_path = skill_dir / "assets/runtime-manifest.json"
        if (manifest_path.is_symlink() or manifest_path.parent.is_symlink() or not manifest_path.is_file()
                or manifest_path.stat().st_size > MAX_MANIFEST_BYTES):
            raise ValueError("missing or invalid bundle manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "binder.runtime-bundle.v1":
            raise ValueError("unsupported bundle manifest")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("empty bundle manifest")
        for relative, expected in files.items():
            relative_path = Path(relative)
            if (relative_path.is_absolute() or not relative_path.parts
                    or any(part in {"..", ".env"} for part in relative_path.parts)
                    or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError("invalid bundle manifest entry: " + relative)
            path = skill_dir / relative_path
            if (any((skill_dir / parent).is_symlink() for parent in (relative_path, *relative_path.parents))
                    or not path.resolve().is_relative_to(skill_dir.resolve()) or not path.is_file()):
                raise ValueError("missing or escaping bundled file: " + relative)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("bundle hash mismatch: " + relative)
        required = {"SKILL.md", "scripts/trex_skill.py", "scripts/binder_runtime/controller.py"}
        if not required.issubset(files):
            raise ValueError("manifest lacks required entrypoints")
        # A stale deployment may leave executable modules from an older bundle.
        # Hashing only declared files would incorrectly certify that deployment.
        for path in (skill_dir / "scripts").rglob("*"):
            if path.is_symlink():
                raise ValueError("symlink in bundled execution files")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                if path.relative_to(skill_dir).as_posix() not in files:
                    raise ValueError("undeclared bundled execution file: " + path.relative_to(skill_dir).as_posix())
        report["bundle"].update(valid=True, upstream_commit=manifest.get("upstream_commit"),
                                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                checked_files=len(files))
        return True
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        note("invalid_bundle", str(exc), "error")
        return False


def inspect_tools(role, report, note):
    tool_names = ["git", "ssh"] if role == "workstation" else [
        "git", "bash", "uv", "sbatch", "squeue", "sacct", "g++", "cmake", "cargo", "curl", "nvcc",
    ]
    for name in tool_names:
        executable = shutil.which(name)
        tool = {"available": executable is not None, "executable": executable}
        report["tools"][name] = tool
        if executable is None:
            note("missing_tool", name + " was not found on this host's PATH.")
            continue
        if name == "cargo":
            # A rustup proxy can sync/install a toolchain even for --version.
            tool["status"] = "presence_only"
            tool["reason"] = "Version execution skipped because rustup proxies may install a toolchain."
            continue
        tool.update(probe([executable, "-V" if name == "ssh" else "--version"]))
        if tool["status"] != "ok":
            note("version_unavailable", name + " is present but its version probe did not complete successfully.")


def inventory(skill_dir, role, asset_root=None):
    """Inventory one host, optionally verifying the standalone skill bundle."""
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "read_only_inventory",
        "host_only": skill_dir is None,
        "production_ready": False,
        "role": role,
        "scope": "This host only; presence and version observations are not runtime validation.",
        "host": {"system": platform.system(), "release": platform.release(),
                 "machine": platform.machine(), "python_version": platform.python_version(),
                 "python_executable": sys.executable},
        "bundle": None if skill_dir is None else {
            **path_inventory(skill_dir), "valid": False},
        "disk": {"cwd": disk_inventory(Path.cwd())} if skill_dir is None else {"bundle": disk_inventory(skill_dir)},
        "tools": {},
        "paths": {},
        "gpu": {"status": "not_checked", "reason": (
            "GPU inventory has not been reached; bundle validation may have stopped this report."
            if role == "remote" else "GPU inventory is only attempted with --role remote.")},
        "observations": [],
    }

    def note(code, message, level="warning"):
        report["observations"].append({"level": level, "code": code, "message": message})

    if skill_dir is not None and not inspect_bundle(skill_dir, report, note):
        return report, 2
    inspect_tools(role, report, note)
    if skill_dir is None:
        note("bundle_not_checked", "Host-only inventory: no skill files were read.", "info")

    if asset_root is not None:
        report["asset_root"] = path_inventory(asset_root)
        report["disk"]["asset_root"] = disk_inventory(asset_root)
        if not asset_root.is_dir():
            note("missing_asset_root", "The supplied asset root is not an existing directory.")
    else:
        report["asset_root"] = None
        note("asset_root_unspecified", "No asset root was supplied; model and asset contents have not been inspected.", "info")

    if role == "remote":
        executable = shutil.which("nvidia-smi")
        if executable:
            report["gpu"] = {"executable": executable, **probe([
                executable, "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ])}
        else:
            report["gpu"] = {"status": "missing", "executable": None}
        if report["gpu"]["status"] != "ok" or not report["gpu"].get("output"):
            note("gpu_unavailable", "No usable GPU inventory was returned. Login nodes may have no GPU; this is a warning only.")
    note("inventory_only", "No environments, packages, assets, GPU computations, or design workflows were validated. Production readiness is not established.", "info")
    return report, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--skill-dir", type=normalized_path,
                        help="Standalone skill directory on this host; verify its bundled file hashes")
    source.add_argument("--host-only", action="store_true",
                        help="Inventory this host without reading skill files")
    parser.add_argument("--role", choices=("workstation", "remote"), default="workstation")
    parser.add_argument("--asset-root", type=normalized_path,
                        help="Optional asset directory on this host (existence and disk space only)")
    args = parser.parse_args(argv)
    report, exit_code = inventory(args.skill_dir, args.role, args.asset_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
