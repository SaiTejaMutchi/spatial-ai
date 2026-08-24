"""Run the frozen pipeline over the frozen CA-1M held-out manifest and aggregate.

Order per capture is fixed and matters: normalize, reconstruct, hash the
prediction, and only then open a ground-truth file. The per-capture record keeps
both timestamps so a reader can check that order was kept rather than trust it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .ca1m_eval import evaluate, sha256_file, utc_now

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "validation" / "manifests" / "ca1m_heldout_manifest.json"


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)


def process_capture(capture_id: str, work_dir: Path, connector_stride: int,
                    geometry_stride: int) -> dict[str, Any]:
    """Mobile inputs only, all the way to a frozen prediction."""
    source = REPO_ROOT / "samples" / "arkitscenes" / "raw" / "Validation" / capture_id
    if not source.is_dir():
        return {"captureId": capture_id, "state": "MISSING_MOBILE_INPUT",
                "detail": f"{source} is absent"}

    normalized = work_dir / capture_id / "nc"
    geometry = work_dir / capture_id / "geom"
    normalized.parent.mkdir(parents=True, exist_ok=True)

    connector = _run([sys.executable, "-m", "pipeline.connectors.cli", str(source),
                      str(normalized), "--stride", str(connector_stride),
                      "--classification", "public_development_fixture", "--no-ai"])
    if connector.returncode != 0:
        return {"captureId": capture_id, "state": "CONNECTOR_FAILURE",
                "detail": connector.stderr.strip()[-500:]}

    reconstruct = _run([sys.executable, "-m", "pipeline.geometry.run", str(normalized),
                        str(geometry), "--frame-stride", str(geometry_stride)])
    model_path = geometry / "spatial_model.json"
    if reconstruct.returncode != 0 or not model_path.is_file():
        return {"captureId": capture_id, "state": "GEOMETRY_FAILURE",
                "detail": (reconstruct.stderr.strip()[-500:] or "no model written")}

    freeze_path = work_dir / capture_id / "prediction_freeze.json"
    freeze_path.write_text(json.dumps({
        "captureId": capture_id,
        "predictionSha256": sha256_file(model_path),
        "predictionFrozenUtc": utc_now(),
        "note": "Written before any CA-1M gt/ asset was opened.",
    }, indent=2) + "\n")

    return {"captureId": capture_id, "state": "PREDICTION_FROZEN",
            "normalized": str(normalized), "model": str(model_path),
            "freeze": str(freeze_path)}


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    heights = [r["roomHeightError"]["absoluteError_cm"] for r in results
               if r.get("roomHeightError")]
    signed = [r["roomHeightError"].get("signedError_cm") for r in results
              if r.get("roomHeightError")
              and r["roomHeightError"].get("signedError_cm") is not None]
    scored = [s for r in results for s in r.get("surfaceDistances", []) if s.get("scored")]

    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {"n": len(values),
                "mean": round(float(array.mean()), 3),
                "median": round(float(np.median(array)), 3),
                "p90": round(float(np.percentile(array, 90)), 3),
                "worst": round(float(array.max()), 3)}

    by_type: dict[str, list[float]] = {}
    for surface in scored:
        by_type.setdefault(surface["type"], []).append(surface["median_cm"])

    gate_passes = sum(1 for h in heights if h <= 1.5)
    return {
        "capturesEvaluated": len(results),
        "roomHeightAbsoluteError_cm": stats(heights),
        "roomHeightSignedError_cm": (
            {**stats(signed),
             "allSameSign": bool(signed) and (all(v < 0 for v in signed)
                                              or all(v > 0 for v in signed)),
             "note": "Sign is kept because a bias and a scatter are different "
                     "defects. A consistent sign across independent captures "
                     "points at the estimator, not at noise."}
            if signed else None),
        "roomHeightGate": {
            "gate": "at most 1.5 cm",
            "passed": gate_passes,
            "failed": len(heights) - gate_passes,
            "passRate": (round(gate_passes / len(heights), 3) if heights else None),
        },
        "surfaceMedianDistance_cm_byType": {t: stats(v) for t, v in sorted(by_type.items())},
        "failures": {
            "surfacesWithoutLaserSupport": sum(
                r.get("failures", {}).get("surfacesWithoutLaserSupport", 0) for r in results),
            "capturesWithoutHeightReference": sum(
                1 for r in results if not r.get("roomHeightError")),
        },
        "coverage": {
            "meanUnregisteredGtFraction": round(float(np.mean(
                [r["coverage"]["unregisteredFraction"] for r in results])), 4) if results else None,
        },
        "alignment": {
            "rotationResidualP90_deg": stats(
                [r["alignment"]["rotationResidualP90_deg"] for r in results]),
            "note": "The mobile trajectory is ARKit odometry and the reference poses are "
                    "registered per frame, so this residual is the capture's own drift. "
                    "Errors at or below this scale are not separable from it.",
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--ca1m-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--connector-stride", type=int, default=6)
    parser.add_argument("--geometry-stride", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "validation" / "results" / "ca1m"
                                / "ca1m_multiscene.json")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)

    prepared, results, skipped = [], [], []
    for capture in manifest["captures"]:
        capture_id = capture["captureId"]
        print(f"[{capture_id}] reconstructing from mobile inputs...", flush=True)
        record = process_capture(capture_id, args.work_dir,
                                 args.connector_stride, args.geometry_stride)
        if record["state"] != "PREDICTION_FROZEN":
            print(f"[{capture_id}] {record['state']}: {record.get('detail','')[:160]}")
            skipped.append(record)
            continue
        prepared.append(record)

    for record in prepared:
        capture_id = record["captureId"]
        print(f"[{capture_id}] scoring against laser ground truth...", flush=True)
        try:
            result = evaluate(Path(record["normalized"]), Path(record["model"]),
                              args.ca1m_root, capture_id, Path(record["freeze"]))
        except Exception as exc:
            print(f"[{capture_id}] EVALUATION_FAILURE: {exc}")
            skipped.append({"captureId": capture_id, "state": "EVALUATION_FAILURE",
                            "detail": str(exc)})
            continue
        results.append(result)

    payload = {
        "generatedUtc": utc_now(),
        "manifestId": manifest["manifestId"],
        "manifestFrozenUtc": manifest["frozenUtc"],
        "geometryConfigHash": manifest["geometryConfigHash"],
        "heldOut": True,
        "claimBoundary": (
            "Geometry was evaluated on held-out real Apple mobile RGB-D captures against "
            "independently laser-scanner-derived ground truth in registered laser space. "
            "This is not a Stray production-path claim, and it validates no opening or "
            "tape-measure gate."),
        "aggregate": _aggregate(results),
        "captures": results,
        "notEvaluated": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    aggregate = payload["aggregate"]
    print(f"\n-> {args.output}")
    print(f"captures evaluated: {aggregate['capturesEvaluated']}, "
          f"not evaluated: {len(skipped)}")
    signed_stats = aggregate.get("roomHeightSignedError_cm")
    if signed_stats and signed_stats.get("allSameSign"):
        print(f"room height signed error is the same sign on every capture: "
              f"median {signed_stats['median']} cm -> a systematic bias, not scatter")
    height = aggregate["roomHeightAbsoluteError_cm"]
    if height:
        print(f"room height |error| cm: median {height['median']}, mean {height['mean']}, "
              f"p90 {height['p90']}, worst {height['worst']}")
        gate = aggregate["roomHeightGate"]
        print(f"1.5 cm gate: {gate['passed']} passed / {gate['failed']} failed")
    for surface_type, stats in aggregate["surfaceMedianDistance_cm_byType"].items():
        if stats:
            print(f"  {surface_type:8s} median-of-medians {stats['median']} cm "
                  f"(worst {stats['worst']} cm, n={stats['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
