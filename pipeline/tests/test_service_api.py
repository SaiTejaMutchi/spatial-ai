"""Local service and integration tests, and the property-and-casualty claims bridge.

The plan is explicit that a CLI success or a static frontend cannot establish
DEV_COMPLETE, so these tests drive the real API: create a scan, process it
through the actual pipeline, and read back the artifacts the UI renders.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_FIXTURE = REPO_ROOT / "samples/arkitscenes/raw/Training/47333462"
FRONTEND = REPO_ROOT / "apps" / "frontend"

requires_fixture = pytest.mark.skipif(
    not PRIMARY_FIXTURE.is_dir(), reason="ARKitScenes fixture not present")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["SPATIAL_AI_STATE_DIR"] = str(tmp_path_factory.mktemp("scans"))
    from pipeline.ai import verifier as verifier_mod

    def fake_assess(self, system, summary, images, response_schema, model):
        return {
            "schemaVersion": "0.1", "status": "completed", "model": model,
            "provider": "groq", "promptVersion": "spatial_verifier_v0.1",
            "generatedAt": "2026-08-23T00:00:00+00:00",
            "roomTypeHypothesis": "bedroom", "findings": [],
            "usage": {"inputTokens": 1, "outputTokens": 1},
        }

    original = verifier_mod.GroqVerifierClient.assess
    verifier_mod.GroqVerifierClient.assess = fake_assess
    from fastapi.testclient import TestClient

    import service.api as api_module
    import importlib
    importlib.reload(api_module)
    try:
        yield TestClient(api_module.app)
    finally:
        verifier_mod.GroqVerifierClient.assess = original


@pytest.fixture(scope="module")
def processed(client):
    if not PRIMARY_FIXTURE.is_dir():
        pytest.skip("fixture not present")
    created = client.post("/api/scans", json={
        "source_path": str(PRIMARY_FIXTURE), "label": "test",
        "classification": "public_development_fixture"}).json()
    client.post(f"/api/scans/{created['scanId']}/process")
    return client.get(f"/api/scans/{created['scanId']}/status").json()


# --------------------------------------------------------------------------
# the service is local, thin, and honest about it
# --------------------------------------------------------------------------

def test_health_states_that_nothing_leaves_the_machine(client):
    body = client.get("/api/health").json()
    assert body["localOnly"] is True
    assert "Nothing is uploaded" in body["note"]
    assert set(body["supportedSources"]) == {"arkitscenes", "stray_scanner", "unity_obj"}


def test_the_api_duplicates_no_pipeline_logic():
    """The boundary that makes the final Stray run a rerun, not a reimplementation."""
    import service.api
    import service.scan_manager

    for module in (service.api, service.scan_manager):
        text = Path(module.__file__).read_text()
        for forbidden in ("fit_plane", "np.histogram", "voxel_downsample",
                          "shoelace", "render_floorplan(model", "def build_envelope"):
            assert forbidden not in text, f"{module.__name__} reimplements {forbidden}"


def test_source_detection_happens_at_intake(client):
    if not PRIMARY_FIXTURE.is_dir():
        pytest.skip("fixture not present")
    created = client.post("/api/scans", json={
        "source_path": str(PRIMARY_FIXTURE)}).json()
    assert created["sourceType"] == "arkitscenes"
    assert created["connector"] == "ARKitScenesConnector"
    stages = {s["name"]: s for s in created["stages"]}
    assert stages["source_detected"]["state"] == "complete"


def test_a_missing_capture_fails_with_a_precise_message(client):
    created = client.post("/api/scans", json={
        "source_path": "/definitely/not/here"}).json()
    assert created["status"] == "failed"
    assert created["failureClass"] == "CONNECTOR_FAILURE"
    assert "does not exist" in created["error"]


def test_quoted_capture_paths_are_stripped_at_intake(client):
    if not PRIMARY_FIXTURE.is_dir():
        pytest.skip("fixture not present")
    quoted = f"'{PRIMARY_FIXTURE}'"
    created = client.post("/api/scans", json={"source_path": quoted}).json()
    assert created["status"] != "failed"
    assert created["sourceType"] == "arkitscenes"
    assert created["error"] is None


def test_the_add_capture_form_strips_wrapping_quotes():
    src = (FRONTEND / "screens.js").read_text()
    assert "function cleanCapturePath" in src
    assert "source_path: cleanCapturePath(sourcePath)" in src
    assert "does not exist" in src


def test_add_capture_uses_native_mac_and_windows_folder_pickers():
    src = (FRONTEND / "screens.js").read_text()
    css = (FRONTEND / "styles.css").read_text()
    app = (FRONTEND / "app.js").read_text()
    assert "webkitdirectory" in src
    assert "Choose Folder" in src
    assert "Select folder" in src
    assert "hostPlatform" in src and "hostPlatform" in app
    assert 'data-platform="mac"' in css
    assert 'data-platform="windows"' in css
    assert "/api/scans/from-folder" in src


def test_a_folder_upload_is_copied_onto_this_machine(client, tmp_path):
    from pipeline.tests import synthetic

    source = synthetic.make_stray(tmp_path / "src", frames=2)
    uploads = []
    paths = []
    for path in source.rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(source.parent)).replace("\\", "/")
            uploads.append(("files", (path.name, path.read_bytes())))
            paths.append(relative)
    created = client.post(
        "/api/scans/from-folder",
        files=uploads,
        data={"classification": "final_private_capture", "paths": paths},
    ).json()
    assert created["sourceType"] == "stray_scanner"
    assert created["status"] != "failed"
    assert created["failureClass"] is None


def test_a_large_folder_is_not_rejected_at_the_multipart_cap(client):
    """A real Stray export exceeds Starlette's default 1000-field form limit."""
    uploads = [("files", (f"{index:04d}.png", b"x")) for index in range(1200)]
    paths = [f"depth/{index:04d}.png" for index in range(1200)]
    response = client.post(
        "/api/scans/from-folder",
        files=uploads,
        data={"classification": "final_private_capture", "paths": paths},
    )
    assert "Too many fields" not in response.text
    assert "Too many files" not in response.text
    assert response.status_code == 200


def test_a_folder_upload_rejects_parent_directory_paths(client):
    response = client.post(
        "/api/scans/from-folder",
        files=[("files", ("passwd", b"no"))],
        data={"paths": "../etc/passwd", "classification": "final_private_capture"},
    )
    assert response.status_code == 400
    assert "invalid upload path" in response.json()["detail"]


def test_stray_uses_a_denser_geometry_stride_than_public_arkit():
    from service.scan_manager import geometry_stride_for_source

    assert geometry_stride_for_source("stray_scanner", 4) == 1
    assert geometry_stride_for_source("arkitscenes", 4) == 4
    assert geometry_stride_for_source("unity_obj", 4) == 4


def test_an_unrecognised_source_is_classified_as_a_connector_failure(client, tmp_path):
    mystery = tmp_path / "mystery"
    mystery.mkdir()
    (mystery / "notes.txt").write_text("hello")
    created = client.post("/api/scans", json={"source_path": str(mystery)}).json()
    assert created["failureClass"] == "CONNECTOR_FAILURE"
    assert "matches no known capture source" in created["error"]


def test_unknown_scan_ids_are_404(client):
    assert client.get("/api/scans/scan-nope/status").status_code == 404
    assert client.post("/api/scans/scan-nope/process").status_code == 404


def test_asking_for_a_model_before_processing_is_a_clear_conflict(client):
    if not PRIMARY_FIXTURE.is_dir():
        pytest.skip("fixture not present")
    created = client.post("/api/scans", json={
        "source_path": str(PRIMARY_FIXTURE)}).json()
    response = client.get(f"/api/scans/{created['scanId']}/model")
    assert response.status_code == 409
    assert "process it first" in response.json()["detail"]


def test_only_declared_artifacts_can_be_downloaded(client, processed):
    response = client.get(f"/api/scans/{processed['scanId']}/artifacts/../state.json")
    assert response.status_code in (404, 400)


# --------------------------------------------------------------------------
# a real end-to-end run
# --------------------------------------------------------------------------

@requires_fixture
def test_every_stage_the_plan_names_is_reported(processed):
    from service.models import STAGES

    assert [s["name"] for s in processed["stages"]] == list(STAGES)


@requires_fixture
def test_processing_completes_through_the_real_pipeline(processed):
    assert processed["status"] == "complete"
    assert processed["failureClass"] is None
    stages = {s["name"]: s for s in processed["stages"]}
    assert stages["normalized_capture"]["state"] == "complete"
    assert "NORMALIZED_CAPTURE_VALID" in stages["normalized_capture"]["detail"]
    assert stages["geometry"]["state"] == "complete"
    assert stages["canonical_model"]["state"] == "complete"
    assert stages["complete"]["state"] == "complete"


@requires_fixture
def test_successful_stages_stay_visible_and_carry_detail(processed):
    for stage in processed["stages"]:
        if stage["state"] == "complete":
            assert stage["detail"], f"{stage['name']} completed with no detail"


@requires_fixture
def test_the_summary_reports_real_measurements(processed):
    summary = processed["summary"]
    assert 2.0 < summary["roomHeight_m"] < 3.5
    assert summary["floorArea_m2"] > 4.0
    assert summary["surfaceCount"] >= 4
    assert summary["modelSha256"]


@requires_fixture
def test_the_model_the_ui_reads_is_valid(client, processed):
    from pipeline.contracts.validate_model import validate_model

    model = client.get(f"/api/scans/{processed['scanId']}/model").json()
    assert validate_model(model) == []


@requires_fixture
def test_every_generated_artifact_downloads(client, processed):
    artifacts = client.get(
        f"/api/scans/{processed['scanId']}/artifacts").json()["artifacts"]
    generated = [a for a in artifacts if a["available"]]
    assert len(generated) >= 8
    for artifact in generated:
        response = client.get(
            f"/api/scans/{processed['scanId']}/artifacts/{artifact['name']}")
        assert response.status_code == 200, artifact["name"]
        assert len(response.content) > 0


@requires_fixture
def test_artifacts_carry_a_development_or_experimental_status(client, processed):
    artifacts = client.get(
        f"/api/scans/{processed['scanId']}/artifacts").json()["artifacts"]
    by_name = {a["name"]: a for a in artifacts}
    assert by_name["spatial_model.json"]["status"] == "development-only"
    assert by_name["loss_preview.json"]["status"] == "experimental"
    assert by_name["benchmark.csv"]["status"] == "pending"
    assert not any(a["status"] == "final-verified" for a in artifacts), (
        "a public fixture must never be labelled final")


@requires_fixture
def test_the_benchmark_stage_is_not_applicable_for_a_fixture(processed):
    stage = next(s for s in processed["stages"] if s["name"] == "benchmark")
    assert stage["state"] == "not_applicable"
    assert "no tape ground truth" in stage["detail"].lower()


@requires_fixture
def test_ai_review_runs_with_operator_approval(processed):
    stage = next(s for s in processed["stages"] if s["name"] == "ai_review")
    assert stage["state"] == "complete"
    assert processed["summary"]["aiStatus"] == "completed"


@requires_fixture
def test_a_second_process_of_the_same_room_reuses_stored_ai(client, processed):
    created = client.post("/api/scans", json={
        "source_path": str(PRIMARY_FIXTURE), "label": "reuse",
        "classification": "public_development_fixture"}).json()
    client.post(f"/api/scans/{created['scanId']}/process")
    model = client.get(f"/api/scans/{created['scanId']}/model").json()
    assert model["aiAssessments"][0]["status"] == "completed"
    assert model["provenance"]["aiReview"]["cacheHit"] is True


@requires_fixture
def test_evidence_frames_are_served_to_the_ui(client, processed):
    model = client.get(f"/api/scans/{processed['scanId']}/model").json()
    for view in model["evidence"]:
        name = view["path"].split("/")[-1]
        response = client.get(f"/api/scans/{processed['scanId']}/evidence/{name}")
        assert response.status_code == 200
        assert response.content[:4] == b"\x89PNG"


# --------------------------------------------------------------------------
# The property-and-casualty claims grounding bridge
# --------------------------------------------------------------------------

@requires_fixture
def test_loss_preview_is_unmistakably_a_development_fixture(client, processed):
    preview = client.get(
        f"/api/scans/{processed['scanId']}/artifacts/loss_preview.json").json()
    assert preview["label"] == "DEVELOPMENT LOSS FIXTURE"
    assert preview["isRealDamageEvidence"] is False
    assert "not damage detection" in preview["statement"]


@requires_fixture
def test_the_proposal_attaches_to_a_real_surface_and_real_evidence(client, processed):
    model = client.get(f"/api/scans/{processed['scanId']}/model").json()
    preview = client.get(
        f"/api/scans/{processed['scanId']}/artifacts/loss_preview.json").json()
    proposal = preview["proposals"][0]
    assert proposal["surfaceId"] in {s["id"] for s in model["surfaces"]}
    evidence_ids = {v["id"] for v in model["evidence"]}
    assert set(proposal["evidenceFrameIds"]) <= evidence_ids
    assert proposal["status"] == "proposed_experimental"
    assert proposal["reviewStatus"] == "human_review_required"


@requires_fixture
def test_any_quantity_is_produced_by_geometry_not_by_ai(client, processed):
    preview = client.get(
        f"/api/scans/{processed['scanId']}/artifacts/loss_preview.json").json()
    proposal = preview["proposals"][0]
    if proposal["registration"]["status"] == "registered":
        assert proposal["quantity"]["producer"] == "geometry_pipeline"
        assert proposal["quantity"]["affectedArea_m2"] > 0
    else:
        assert "quantity" not in proposal


@requires_fixture
def test_unresolved_registration_emits_no_quantity():
    """The rule that stops an ungrounded number being reported."""
    from pipeline.loss_preview.preview import register_region_to_surface

    surface = {"id": "wall-001", "plane": {"normal": [0.0, 0.0, 0.0], "offset_m": 0.0}}
    registration = register_region_to_surface(
        {"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}, surface,
        __import__("numpy").eye(4),
        {"width": 256, "height": 192, "fx": 200, "fy": 200, "cx": 128, "cy": 96})
    assert registration["status"] == "unresolved"
    assert "affectedArea_m2" not in registration
    assert "no fitted plane" in registration["reason"]


@requires_fixture
def test_synthetic_content_never_reaches_production_arrays(client, processed):
    model = client.get(f"/api/scans/{processed['scanId']}/model").json()
    assert model["damage"] == []
    assert model["scope"] == []
    for surface in model["surfaces"]:
        assert surface["damage"] == []


# --------------------------------------------------------------------------
# library: saved results, reopening, and renaming
# --------------------------------------------------------------------------

@requires_fixture
def test_the_library_returns_user_facing_records(client, processed):
    scans = client.get("/api/scans").json()["scans"]
    assert scans
    record = next(s for s in scans if s["id"] == processed["scanId"])
    for key in ("id", "name", "createdAt", "status", "classification",
                "thumbnailArtifact", "summary", "authenticity"):
        assert key in record, key
    assert record["thumbnailArtifact"] == "floorplan.svg"
    assert record["authenticity"]["kind"] == "published_dataset"
    assert "tape-measure" in record["authenticity"]["cite"]
    assert "FARO correspondence" in record["authenticity"]["cite"]
    assert "no accuracy is written" in record["authenticity"]["cite"]


def test_an_explicit_final_capture_is_not_relabelled_as_a_repo_sample():
    from service.scan_manager import describe_authenticity

    record = describe_authenticity({
        "classification": "final_private_capture",
        "sourceType": "stray_scanner",
        "sourcePath": "/Users/me/Repo/samples/iphone-actual/e30fe3cae4",
    })
    assert record["kind"] == "local_capture"
    assert "Not a private iPhone" not in record["cite"]


def test_a_scan_can_be_deleted(client, tmp_path):
    mystery = tmp_path / "mystery"
    mystery.mkdir()
    (mystery / "notes.txt").write_text("hello")
    created = client.post("/api/scans", json={"source_path": str(mystery)}).json()
    scan_id = created["scanId"]
    assert client.delete(f"/api/scans/{scan_id}").status_code == 200
    assert client.get(f"/api/scans/{scan_id}").status_code == 404
    assert client.delete("/api/scans/scan-nope").status_code == 404


def test_the_library_cards_cite_source_authenticity_and_can_be_deleted():
    src = (FRONTEND / "screens.js").read_text()
    css = (FRONTEND / "styles.css").read_text()
    assert "authenticity.cite" in src
    assert "Delete" in src
    assert "View sample" in src
    assert "api.delete" in src
    assert "isDatasetTile" in src
    assert "public_review_example" in src
    assert "Tape accuracy is not attached" in src
    assert "FARO correspondence" in src
    assert "no accuracy is written" in src
    assert 'data-kind="published_dataset"' in css
    assert 'data-kind="public_sample"' in css
    assert 'data-kind="local_capture"' in css


def test_add_capture_explains_iphone_lidar_export_and_later_damage():
    src = (FRONTEND / "screens.js").read_text()
    assert "iPhone 13 Pro Max" in src
    assert "iPhone 12 Pro" in src
    assert "odometry.csv" in src
    assert "depth/000000.png" in src
    assert "rgb.mp4" in src
    assert "Choose Zip" in src
    assert "damage[]" in src
    assert "Later — not in this pass" in src
    assert "nothing is uploaded" in src.lower()


def test_a_zip_of_a_stray_folder_is_accepted(client, tmp_path):
    import io
    import zipfile

    from pipeline.tests import synthetic

    source = synthetic.make_stray(tmp_path / "src", frames=2)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent))
    created = client.post(
        "/api/scans/from-folder",
        files=[("files", ("capture.zip", buffer.getvalue(), "application/zip"))],
        data={"classification": "final_private_capture", "paths": ["capture.zip"]},
    ).json()
    assert created["sourceType"] == "stray_scanner"
    assert created["status"] != "failed"


@requires_fixture
def test_library_summary_counts_come_from_real_artifacts(client, processed):
    """Counts are read back from generated files, not cached at process time."""
    record = client.get(f"/api/scans/{processed['scanId']}").json()
    summary = record["summary"]
    model = client.get(f"/api/scans/{processed['scanId']}/model").json()
    assessment = model["aiAssessments"][0]

    expected = len(assessment.get("findings", []))
    condition = client.get(
        f"/api/scans/{processed['scanId']}/artifacts/visible_condition.json")
    if condition.status_code == 200 and condition.json().get("proposal"):
        expected += 1
    assert summary["aiFindingCount"] == expected
    assert summary["needsReviewCount"] <= summary["aiFindingCount"]
    assert summary["surfaceCount"] == len(model["surfaces"])


@requires_fixture
def test_reopening_a_saved_scan_does_not_reprocess_it(client, processed):
    before = client.get(f"/api/scans/{processed['scanId']}").json()
    again = client.get(f"/api/scans/{processed['scanId']}").json()
    assert before["updatedAt"] == again["updatedAt"]
    assert again["status"] == "complete"


@requires_fixture
def test_a_saved_scan_can_be_renamed(client, processed):
    renamed = client.patch(f"/api/scans/{processed['scanId']}",
                           json={"name": "  Front bedroom  "}).json()
    assert renamed["name"] == "Front bedroom"
    assert client.get(f"/api/scans/{processed['scanId']}").json()["name"] == "Front bedroom"
    client.patch(f"/api/scans/{processed['scanId']}", json={"name": "test"})


def test_renaming_an_unknown_scan_is_404(client):
    assert client.patch("/api/scans/scan-nope", json={"name": "x"}).status_code == 404


def test_an_empty_name_is_rejected(client, processed):
    response = client.patch(f"/api/scans/{processed['scanId']}", json={"name": "   "})
    assert response.status_code in (400, 422)


def test_client_routes_serve_the_app_shell(client):
    """A deep link or refresh must land on the app, not a 404."""
    response = client.get("/scans/scan-anything")
    assert response.status_code == 200
    assert "<title>Spatial AI</title>" in response.text


def test_app_assets_are_never_cached(client):
    """A stale stylesheet silently shows the wrong product."""
    assert "no-store" in client.get("/styles.css").headers.get("cache-control", "")
    assert "no-store" in client.get("/app.js").headers.get("cache-control", "")


def test_asset_links_are_absolute_so_deep_links_work():
    """Relative asset URLs resolve under /scans/ and return HTML instead of CSS."""
    markup = (FRONTEND / "index.html").read_text()
    assert 'href="/styles.css' in markup
    assert 'src="/app.js' in markup


# --------------------------------------------------------------------------
# the 3D viewer and DOM helpers
# --------------------------------------------------------------------------

def test_conditional_children_never_reach_a_raw_append():
    """append() stringifies null, so a conditional child renders as "null" text.

    h(), mount() and remount() filter; the raw DOM calls do not. This is the
    lint that keeps the literal word "null" out of the interface.
    """
    import re

    offenders = []
    for path in sorted(FRONTEND.glob("*.js")):
        if path.name == "lib.js":
            continue  # the helpers themselves are where the filtering lives
        src = path.read_text()
        for match in re.finditer(r"\.(append|replaceChildren)\(", src):
            index, depth = match.end(), 1
            while index < len(src) and depth:
                depth += (src[index] == "(") - (src[index] == ")")
                index += 1
            args = src[match.end():index - 1]
            if re.search(r":\s*(null|undefined|false)\b", args) or "&&" in args:
                line = src[:match.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line}")
    assert not offenders, (
        "use mount()/remount() for conditional children: " + ", ".join(offenders))


def test_the_three_runtime_is_vendored_and_pinned():
    """No CDN: a review tool that phones out is not local-only."""
    markup = (FRONTEND / "index.html").read_text()
    assert "cdn" not in markup.lower()
    assert '"three": "/vendor/three/three.module.js"' in markup
    assert '"three/addons/": "/vendor/three/addons/"' in markup
    assert (FRONTEND / "vendor" / "three" / "three.module.js").exists()
    assert (FRONTEND / "vendor" / "three" / "README.md").exists()


def test_the_three_viewer_reads_the_artifact_api():
    """The 3D view renders the pipeline's own OBJ, not a copy shipped alongside."""
    src = (FRONTEND / "three-viewer.js").read_text()
    assert "/api/scans/" in src
    assert "room_model.obj" in src and "room_model.mtl" in src
    assert not list(FRONTEND.rglob("*.obj")), "no OBJ may ship with the frontend"


def test_the_three_viewer_is_keyboard_reachable():
    src = (FRONTEND / "three-viewer.js").read_text()
    assert "tabIndex = 0" in src
    assert "keydown" in src and "ArrowLeft" in src
    assert "aria-label" in src


def test_the_room_can_be_seen_into():
    """A room is a closed box; without the cutaway the 3D tab shows a solid."""
    src = (FRONTEND / "three-viewer.js").read_text()
    assert "updateCutaway" in src
    assert "cutaway" in (FRONTEND / "result.js").read_text().lower()


@requires_fixture
def test_the_obj_exposes_every_canonical_surface_id(client, processed):
    """3D selection depends on the exporter keeping the IDs the inspector uses."""
    scan_id = processed["scanId"]
    obj = client.get(f"/api/scans/{scan_id}/artifacts/room_model.obj").text
    entity_map = client.get(
        f"/api/scans/{scan_id}/artifacts/room_model_entity_map.json").json()
    model = client.get(f"/api/scans/{scan_id}/model").json()

    surface_ids = {surface["id"] for surface in model["surfaces"]}
    assert surface_ids, "no surfaces to select"
    assert surface_ids == set(entity_map["surfaces"]), "entity map drifted from the model"
    groups = {line.split(maxsplit=1)[1].strip()
              for line in obj.splitlines() if line.startswith("g ")}
    assert surface_ids <= groups, "an OBJ group is missing a canonical surface ID"


# --------------------------------------------------------------------------
# the frontend renders API data, not committed results
# --------------------------------------------------------------------------

def _frontend_sources() -> dict[str, str]:
    """Every shipped frontend file, so a split into modules cannot hide a value."""
    return {
        path.name: path.read_text()
        for path in sorted(FRONTEND.iterdir())
        if path.suffix in {".js", ".html", ".css"}
    }


def test_the_frontend_hardcodes_no_result():
    sources = _frontend_sources()
    assert sources, "no frontend sources found"
    for name, text in sources.items():
        for forbidden in ("47333462", "41418135", "2.61", "16.00", "6.675"):
            assert forbidden not in text, f"{name} hardcodes {forbidden}"


def test_the_frontend_calls_the_real_api():
    combined = "\n".join(_frontend_sources().values())
    assert "/api/scans" in combined, "the frontend must call the real API"
    assert "/api/scans/${" in combined or "/api/scans/" in combined


def test_the_frontend_states_that_processing_is_local():
    combined = "\n".join(_frontend_sources().values())
    assert "nothing is uploaded" in combined.lower()


def test_process_capture_shows_loading_before_the_folder_upload():
    """The progress screen must paint before FormData / POST block the thread."""
    src = (FRONTEND / "screens.js").read_text()
    assert "function presentProcessing(" in src
    assert "function yieldToPaint(" in src
    assert "enterProcessing()" in src
    enter = src.index("const view = enterProcessing();")
    yield_at = src.index("await yieldToPaint();", enter)
    form = src.index("const form = new FormData();", yield_at)
    post = src.index("api.postForm('/api/scans/from-folder'", form)
    assert enter < yield_at < form < post
    assert "Starting…" in src
    assert "paintPending()" in src


def test_the_frontend_labels_ai_and_geometry_producers_separately():
    """The one distinction the interface must make without anyone explaining it."""
    combined = "\n".join(_frontend_sources().values())
    assert "AI interpretation" in combined
    assert "produced by geometry" in combined.lower()
    assert "Geometry measurement" in combined


def test_result_tabs_describe_damage_intelligence_and_drop_benchmark_loss():
    src = (FRONTEND / "result.js").read_text()
    assert "['damage', 'Damage intelligence']" in src
    assert "['loss', 'Loss preview']" not in src
    assert "['benchmark', 'Benchmark']" not in src
    assert "function paintLoss" not in src
    assert "function paintBenchmark" not in src
    assert "Track B and Track C are not implemented" in src
    assert "AI identifies what may be damaged" in src
    assert "damage[]" in src and "scope[]" in src
    assert "Deferred — not evaluated" in src
    assert "Not implemented" in src
    assert "Week 3" in src and "Week 4" in src
    assert "Development sample" in src
    assert "Final capture" in src


def test_ai_review_states_what_it_is_and_what_openings_are():
    src = (FRONTEND / "result.js").read_text()
    assert "AI looks at the photographs" in src
    assert "does not change any measurement" in src
    assert "Doorways and windows" in src
    assert "crop of a geometry gap" in src
    assert "nominated region" in src
    assert "Crop corroborated" in src
    assert "How to read this" not in src
    assert "Show the whole space" in src
    assert "Needs a look" in src
    assert "Paste the Qwen review" in src
    assert "Store review" in src
    assert "Asking Groq" in src
    assert "function requestGroqReview(" in src
    assert "api.post(`/api/scans/${state.scanId}/ai-review`)" in src
    assert "/api/scans/${state.scanId}/ai-review/import" in src
    assert "/api/scans/${state.scanId}/ai-review" in src
    ask = src.index("void requestGroqReview(state)")
    after_ask = src.index("if (phase !== 'done')")
    paste = src.index("paintQwenPaste(state, body)", after_ask)
    assert ask < paste, "Groq must be asked before the Qwen paste is shown"
    assert "Try AI review again" not in src


def test_the_frontend_labels_development_samples():
    combined = "\n".join(_frontend_sources().values())
    assert "Development sample" in combined
    assert "Final capture" in combined


def test_the_frontend_is_installable_as_a_pwa():
    manifest = json.loads((FRONTEND / "manifest.webmanifest").read_text())
    assert manifest["display"] == "standalone"
    assert manifest["name"]
