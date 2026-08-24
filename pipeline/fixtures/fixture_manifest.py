"""Build and verify manifests for immutable public development fixtures.

This module's scope is fixture identity, provenance, integrity, and
readability. It must not perform coordinate normalization, geometry, or any
source-neutral `normalized_capture` work; that belongs to the capture and
normalization stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "samples" / "arkitscenes" / "fixture_manifest.json"

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def digest_tree(root: Path) -> dict:
    """Order-independent digest over a directory's relative paths, sizes, contents."""
    if not root.exists():
        return {"present": False, "file_count": 0, "total_bytes": 0, "tree_sha256": None}
    files = sorted(p for p in root.rglob("*") if p.is_file())
    outer = hashlib.sha256()
    total = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total += size
        outer.update(rel.encode())
        outer.update(str(size).encode())
        outer.update(sha256_file(path).encode())
    return {
        "present": True,
        "file_count": len(files),
        "total_bytes": total,
        "tree_sha256": outer.hexdigest(),
    }


def digest_entry(path: Path) -> dict:
    if path.is_dir():
        return digest_tree(path)
    if not path.exists():
        return {"present": False, "file_count": 0, "total_bytes": 0, "tree_sha256": None}
    return {
        "present": True,
        "file_count": 1,
        "total_bytes": path.stat().st_size,
        "tree_sha256": sha256_file(path),
    }


def build(spec_path: Path, manifest_path: Path) -> dict:
    spec = json.loads(spec_path.read_text())
    manifest = {
        "manifest_version": "0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "pipeline/fixtures/fixture_manifest.py",
        "classification": spec["classification"],
        "dataset": spec["dataset"],
        "fixtures": [],
    }
    for fixture in spec["fixtures"]:
        entry = {k: v for k, v in fixture.items() if k != "assets"}
        entry["assets"] = {}
        for name, rel in fixture["assets"].items():
            target = REPO_ROOT / rel
            record = digest_entry(target)
            record["path"] = rel
            record["source_url"] = fixture["asset_urls"].get(name)
            entry["assets"][name] = record
        # Some assets are listed both individually and inside scene_dir for
        # explicit provenance; count each path once.
        paths = sorted(a["path"] for a in entry["assets"].values() if a["present"])
        unique = [q for q in paths if not any(q != o and q.startswith(o + "/") for o in paths)]
        entry["total_bytes"] = sum(
            a["total_bytes"] for a in entry["assets"].values() if a["path"] in unique
        )
        manifest["fixtures"].append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    failures = 0
    for fixture in manifest["fixtures"]:
        print(f"[{fixture['role']}] {fixture['scene_id']}")
        for name, record in fixture["assets"].items():
            target = REPO_ROOT / record["path"]
            if not record["present"]:
                print(f"  - {name}: RECORDED ABSENT (skipped)")
                continue
            actual = digest_entry(target)
            ok = (
                actual["present"]
                and actual["tree_sha256"] == record["tree_sha256"]
                and actual["file_count"] == record["file_count"]
                and actual["total_bytes"] == record["total_bytes"]
            )
            status = "OK" if ok else "MISMATCH"
            if not ok:
                failures += 1
            print(
                f"  - {name}: {status} files={actual['file_count']} "
                f"bytes={actual['total_bytes']}"
            )
    print("VERIFY:", "PASS" if failures == 0 else f"FAIL ({failures} asset mismatches)")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument("--spec", type=Path, default=REPO_ROOT / "samples" / "arkitscenes" / "fixture_spec.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    if args.command == "build":
        manifest = build(args.spec, args.manifest)
        total = sum(f["total_bytes"] for f in manifest["fixtures"])
        print(f"wrote {args.manifest} ({len(manifest['fixtures'])} fixtures, {total/1e9:.2f} GB)")
        return 0
    return verify(args.manifest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
