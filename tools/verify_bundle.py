#!/usr/bin/env python3
"""Verify the published skill and ZIP using only Python's standard library.

This is a maintainer integrity check, not a GPU environment or design test.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / '.agents/skills/trex-binder-design'
MANIFEST = 'assets/runtime-manifest.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    manifest_bytes = (SKILL / MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest['schema_version'] == 'binder.runtime-bundle.v1', 'Unsupported manifest schema')
    files = manifest['files']
    expected = {}
    for name, checksum in files.items():
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,
                f'Invalid manifest path: {name}')
        path = SKILL / name
        require(path.is_file() and not path.is_symlink(), f'Missing or unsafe file: {name}')
        expected[name] = path.read_bytes()
        require(digest(expected[name]) == checksum, f'Hash mismatch: {name}')
    expected[MANIFEST] = manifest_bytes
    version = manifest['skill_version']
    package = ROOT / 'dist' / f'trex-binder-design-v{version}.zip'
    checksum = package.with_suffix('.zip.sha256').read_text().split()[0]
    require(digest(package.read_bytes()) == checksum, 'Release ZIP checksum mismatch')
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        prefix = 'trex-binder-design/'
        require(len(names) == len(set(names)), 'Duplicate ZIP entries')
        require(set(names) == {prefix + name for name in expected}, 'Unexpected ZIP contents')
        for name, content in expected.items():
            require(archive.read(prefix + name) == content, f'ZIP differs from source: {name}')
    print(json.dumps({
        'ok': True, 'version': version,
        'managed_files': len(files), 'delivered_files': len(expected),
        'zip_sha256': checksum,
        'gpu_environment_checked': False,
    }, indent=2))


if __name__ == '__main__':
    main()
