"""Localhost-only orchestration API.

Thin by design. Every route below either reads scan state from the filesystem or
asks `ScanManager` to invoke pipeline modules. No geometry, no rendering, no
benchmark arithmetic, and no AI logic lives here — that boundary is what makes
the eventual Stray run a rerun of the same code rather than a second one.

No accounts, no authentication, no queue, no cloud. A local SQLite catalog
records completed AI reviews so they are not rerun. Files stay on this machine
and the UI says so.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.connectors.detect import CONNECTORS

from .models import CreateScanRequest, RenameScanRequest, ScanStatusResponse
from .scan_manager import ScanManager

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "apps" / "frontend"

app = FastAPI(title="Spatial AI — local processing service",
              description="Localhost REST API for Spatial AI.",
              version="0.1")

# The PWA is served from this same origin in normal use. CORS is opened only to
# localhost so a separately served dev frontend can still talk to it.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"], allow_headers=["*"],
)

# The storage root can be overridden for testing via SPATIAL_AI_STATE_DIR.
DEFAULT_STATE_DIR = Path(__file__).resolve().parents[1] / "output" / "scans"
manager = ScanManager(Path(os.environ.get("SPATIAL_AI_STATE_DIR", DEFAULT_STATE_DIR)))


@app.middleware("http")
async def no_store_for_app_assets(request, call_next):
    """Never let a browser cache the local UI.

    This is a POC served from disk and edited in place; a stale stylesheet or
    module silently shows the wrong product, which is far more expensive here
    than re-reading a few kilobytes from localhost.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


def _status_response(state: dict) -> ScanStatusResponse:
    return ScanStatusResponse(**{
        "scanId": state["scanId"],
        "status": state["status"],
        "sourceType": state["sourceType"],
        "connector": state["connector"],
        "classification": state["classification"],
        "label": state["label"],
        "stages": state["stages"],
        "currentStage": state.get("currentStage"),
        "failureClass": state.get("failureClass"),
        "error": state.get("error"),
        "createdAt": state["createdAt"],
        "updatedAt": state["updatedAt"],
        "summary": state.get("summary", {}),
    })


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "localOnly": True,
        "note": "Files stay on this machine. Nothing is uploaded to any service.",
        "supportedSources": sorted(CONNECTORS),
    }


@app.get("/api/fixtures")
def fixtures() -> dict:
    """Development fixtures discoverable on this machine, for the picker."""
    root = REPO_ROOT / "samples" / "arkitscenes" / "raw"
    found = []
    if root.is_dir():
        for scene in sorted(root.glob("*/*")):
            if scene.is_dir() and (scene / "lowres_wide.traj").is_file():
                found.append({
                    "id": scene.name,
                    "path": str(scene),
                    "label": f"ARKitScenes {scene.name}",
                    "classification": "public_development_fixture",
                })
    return {"fixtures": found}


@app.get("/api/scans")
def list_scans() -> dict:
    """Library records: user-facing summaries, not pipeline state."""
    return {"scans": [manager.library_record(s) for s in manager.list_scans()]}


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: str) -> dict:
    try:
        return manager.library_record(manager.get(scan_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")


@app.delete("/api/scans/{scan_id}")
def delete_scan(scan_id: str) -> dict:
    try:
        manager.delete(scan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": scan_id}


@app.patch("/api/scans/{scan_id}")
def rename_scan(scan_id: str, request: RenameScanRequest) -> dict:
    try:
        return manager.library_record(manager.rename(scan_id, request.name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/scans")
def create_scan(request: CreateScanRequest) -> ScanStatusResponse:
    state = manager.create(request.source_path, request.label, request.classification)
    return _status_response(state)


@app.post("/api/scans/from-folder")
async def create_scan_from_folder(request: Request) -> ScanStatusResponse:
    """Accept a folder chosen in the native Mac or Windows picker.

    A Stray export can be thousands of depth and confidence frames. Starlette
    defaults to 1000 multipart fields, which rejects that folder. The raised
    cap is local-only — bytes are never forwarded.
    """
    try:
        form = await request.form(max_files=100_000, max_fields=100_000)
    except Exception as exc:  # noqa: BLE001 — surface the parser message
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        uploads = [item for item in form.getlist("files") if hasattr(item, "file")]
        raw_paths = form.getlist("paths")
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        paths = [str(item) for item in raw_paths]
        label = form.get("label")
        classification = form.get("classification") or "final_private_capture"
        if not isinstance(label, str):
            label = None
        if not isinstance(classification, str):
            classification = "final_private_capture"
        if not uploads:
            raise HTTPException(status_code=400, detail="the folder did not contain any files")
        pairs = []
        for index, upload in enumerate(uploads):
            relative = paths[index] if index < len(paths) else ""
            relative = relative or getattr(upload, "filename", None) or "unnamed"
            pairs.append((relative, upload.file))
        try:
            state = manager.create_from_upload(pairs, label, classification)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _status_response(state)
    finally:
        await form.close()


@app.post("/api/scans/{scan_id}/process")
def process_scan(scan_id: str) -> ScanStatusResponse:
    try:
        manager.get(scan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")
    return _status_response(manager.process(scan_id))


@app.post("/api/scans/{scan_id}/ai-review/import")
async def import_ai_review(scan_id: str, request: Request) -> ScanStatusResponse:
    """Store an offline Qwen JSON review. Groq is not called."""
    try:
        manager.get(scan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")
    try:
        raw = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    if isinstance(raw, str):
        raw = _parse_pasted_review(raw)
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    try:
        return _status_response(manager.import_ai_review(scan_id, raw))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _parse_pasted_review(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


@app.post("/api/scans/{scan_id}/ai-review")
async def rerun_ai_review(scan_id: str, request: Request) -> ScanStatusResponse:
    """Store a pasted review, or retry the approved model. Geometry is not rebuilt."""
    try:
        manager.get(scan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")
    raw = None
    content_type = request.headers.get("content-type") or ""
    if "json" in content_type:
        try:
            raw = await request.json()
        except Exception:
            raw = None
    if isinstance(raw, dict) and (
            raw.get("findings") is not None or raw.get("schemaVersion") == "0.1"):
        try:
            return _status_response(manager.import_ai_review(scan_id, raw))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return _status_response(manager.rerun_ai_review(scan_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/scans/{scan_id}/status")
def scan_status(scan_id: str) -> ScanStatusResponse:
    try:
        return _status_response(manager.get(scan_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")


@app.get("/api/scans/{scan_id}/model")
def scan_model(scan_id: str) -> JSONResponse:
    try:
        return JSONResponse(manager.model(scan_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")
    except FileNotFoundError:
        raise HTTPException(
            status_code=409,
            detail="this scan has not produced a spatial model yet; process it first")


@app.get("/api/scans/{scan_id}/artifacts")
def scan_artifacts(scan_id: str) -> dict:
    try:
        manager.get(scan_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no scan '{scan_id}'")
    return {"scanId": scan_id,
            "artifacts": [a.model_dump() for a in manager.artifacts(scan_id)]}


@app.get("/api/scans/{scan_id}/artifacts/{name}")
def scan_artifact(scan_id: str, name: str) -> FileResponse:
    try:
        return FileResponse(manager.artifact_path(scan_id, name), filename=name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{name}' is not an exported artifact")
    except FileNotFoundError:
        raise HTTPException(status_code=404,
                            detail=f"'{name}' has not been generated for this scan")


@app.get("/api/scans/{scan_id}/evidence/{name}")
def scan_evidence(scan_id: str, name: str) -> FileResponse:
    try:
        return FileResponse(manager.evidence_path(scan_id, name))
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid evidence name")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"no evidence frame '{name}'")


if FRONTEND_DIR.is_dir():
    # Client-side routes such as /scans/{id} must serve the app shell rather than
    # 404, so a deep link or a browser refresh lands on the same screen.
    @app.get("/scans/{path:path}", include_in_schema=False)
    def spa_routes(path: str) -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
