# Spatial AI — Status Report

Generated 2026-08-29. Every number below is traceable to a file on disk. Sources are named inline.

Data root for all paths below: `/Users/saitejamutchi/Desktop/Repos/spatial-ai`
(this report and `scripts/recover_scenes_6_10.py` live in the
`claude/safe-cleanup-crash-diagnosis-eada42` worktree; the evaluation data lives only
in the main checkout).

---

## 1. Current N

**N = 10.**

`pipeline/eval/results/latest_run.json` (6,073 bytes, written 2026-08-29 13:11 local /
`2026-08-29T17:11:07Z`) contains 10 entries under `sceneResults`, each carrying
`refHeight_m`, `predHeight_m`, `absError_cm`, `signedError_cm` and `passedGate1_5cm`.
`skippedScenes` is empty.

Verification performed for this report (`scripts/recover_scenes_6_10.py`): for all 10
scenes the predicted height was independently re-derived from
`outputs/dev_<sceneId>/spatial_model.json` (the `room_height` entry in `measurements`)
and compared against `latest_run.json`. **10 of 10 matched to within 0.01 cm.**
Outputs: `pipeline/eval/results/recover_scenes_6_10.json` and
`pipeline/eval/results/verify_scenes_1_5.json`.

Two qualifications on the word "verified":

- **Ground truth is not currently re-derivable.** `cache/arkitscenes/` is empty (0 B) and
  `samples/arkitscenes/raw/` is empty (0 B). No laser `.ply` file for any of the 10 scenes
  remains on disk, so `refHeight_m` for all 10 scenes is carried from the recorded run,
  not recomputed. This applies equally to scenes 1-5 and 6-10.
- **Nothing is committed.** `git ls-files pipeline/eval` returns 0 files — the entire
  evaluation harness (`extract_gt.py`, `evaluator.py`, `baselines.py`, all 74 result
  files including `latest_run.json`) is untracked in git. `outputs/` is git-ignored by
  `.gitignore:25`. So the strict count of scenes with **committed** ground truth +
  prediction + error is **0**; the count of scenes with ground truth + prediction + error
  present on disk and internally consistent is **10**.

### Aggregates, per `latest_run.json`

| Metric | Value |
|---|---|
| Height error mean | 2.8237 cm |
| Height error median | 1.9465 cm |
| Std | 2.3219 cm |
| p90 | 5.694 cm |
| Min / max | 0.651 cm / 8.519 cm |
| Mean 95% CI (bootstrap) | 1.5687 – 4.3825 cm |
| Gate pass rate (≤1.5 cm) | 0.3 (3 of 10) |
| Size buckets | small n=6, medium n=4, large n=0 |

Bias-hypothesis regression: slope −13.0973 cm/m, 95% CI [−32.392, 6.1975], r² 0.2345,
p = 0.1561 — not significant at n=10. The sensitivity table in the same file puts the
required n at 24–77 scenes depending on the assumed true r².

---

## 2. Scenes 1-5 — the fully verified result

| # | sceneId | visit_id | refHeight_m | predHeight_m | absError_cm | gate ≤1.5cm |
|---|---|---|---|---|---|---|
| 1 | 47333462 | 467138 | 2.63 | 2.5994 | 3.063 | no |
| 2 | 41418135 | 416418 | 2.4302 | 2.4121 | 1.807 | no |
| 3 | 41418155 | 416407 | 2.4303 | 2.4096 | 2.065 | no |
| 4 | 41418140 | 416411 | 2.4428 | 2.4245 | 1.828 | no |
| 5 | 42444474 | 421069 | 2.3249 | 2.3102 | 1.472 | yes |

Range **1.47 cm – 3.06 cm**, matching the previously reported five-scene band. All five
predictions re-derived from disk and confirmed against `latest_run.json`
(`verify_scenes_1_5.json`, `matchesPriorRun: true` on all five).

---

## 3. Scenes 6-10 — what happened

All five are **partially recovered**: prediction and error re-derived and confirmed from
files still on disk; ground truth carried from the recorded run because the laser point
clouds are gone.

| # | sceneId | visit_id | refHeight_m | predHeight_m | absError_cm | status |
|---|---|---|---|---|---|---|
| 6 | 42444499 | 421065 | 2.2944 | 2.285349 | 0.905 | partially recovered |
| 7 | 42444511 | 421063 | 2.3073 | 2.329629 | 2.233 | partially recovered |
| 8 | 42444514 | 421061 | 2.1311 | 2.124589 | 0.651 | partially recovered |
| 9 | 42444519 | 421060 | 2.2972 | 2.354144 | 5.694 | partially recovered |
| 10 | 42444574 | 421062 | 2.4578 | 2.372613 | 8.519 | partially recovered |

One sentence each:

- **42444499** — recovered: `outputs/dev_42444499/spatial_model.json` survives and its
  `room_height` of 2.285349 m reproduces the recorded 0.905 cm error exactly; its laser
  `.ply` under visit 421065 is gone, so ground truth 2.2944 m is carried forward, not recomputed.
- **42444511** — same: prediction 2.329629 m reproduced from disk, error 2.233 cm confirmed,
  ground truth 2.3073 m carried forward with the `.ply` for visit 421063 gone.
- **42444514** — same: prediction 2.124589 m reproduced, error 0.651 cm confirmed (gate pass),
  ground truth 2.1311 m carried forward with the `.ply` for visit 421061 gone.
- **42444519** — same: prediction 2.354144 m reproduced, error 5.694 cm confirmed,
  ground truth 2.2972 m carried forward with the `.ply` for visit 421060 gone.
- **42444574** — same, and it was the scene in flight at the second shutdown
  (`outputs/dev_42444574` holds only `spatial_model.json` and `geometry_diagnostics.json`,
  timestamped 13:10, versus the full 11-file layout the earlier scenes have); its prediction
  2.372613 m and 8.519 cm error still reproduce exactly.

Nothing in scenes 6-10 was lost outright. What was lost is the ability to **re-verify**
their ground truth locally without re-downloading the laser scans.

---

## 4. Four-stage progress path

| Stage | State |
|---|---|
| **1 — trustworthy evaluator, N≥1** | **Closed.** `pipeline/eval/extract_gt.py` performs real PLY parsing with SVD floor/ceiling plane fitting; 10 scenes have end-to-end ground truth, prediction and error. |
| **2 — scale to N≥30, baselines** | **In progress. N = 10 of 30 → 33.3%.** A baseline comparison exists at `pipeline/eval/results/baseline_comparison.json` (1,661 bytes). Scenes 11-15 are enumerated but unrun in `scratch/batch_run_scenes.py` (`REMAINING_SCENES`: 42444946, 42444966, 42445021, 42445028, 42445429). |
| **3 — benchmark entry** | **Not started.** |
| **4 — publish** | **Not started.** |

Blocking Stage 2 alongside raw N: the harness itself is uncommitted (see §1), so the
current result set has no durable provenance in git.

---

## 5. Infrastructure notes for the migration

**(a) Local memory and disk limits were hit twice.** Two force-shutdowns originated in this
repo's data handling. The measured cause is bulk intermediate data, not the result files:
before cleanup, `outputs/` held 3.3 GB, of which effectively all of it sat in nine
`outputs/nc_*` normalized-capture directories (186 MB – 618 MB each, extracted RGB, depth
and confidence frames), against `outputs/dev_*` result directories of 56 KB – 324 KB each.
Every file in `pipeline/eval/results/` is ≤ 6,092 bytes, so results were never the problem.
Cleanup on 2026-08-29 removed all `nc_*` directories: `outputs/` went 3.3 GB → 2.9 MB and
free space went 85 GiB → 89 GiB. Work moves to a dedicated cloud VM next, sized so that
per-scene frame extraction is not competing with an IDE, a browser and an agent for the
same RAM.

**(b) All large-file operations go through standalone scripts, everywhere.** Regardless of
where the work runs, any operation on a `.ply`, a `.zip` or an extracted frame directory
must be a single subprocess call to a standalone script that does its work internally and
prints only a short summary — never raw bytes, point arrays or full file contents back
through an agent or IDE context. `scripts/recover_scenes_6_10.py` is the reference shape:
it loads point clouds and JSON documents inside the process, writes its full output to a
file, and emits five lines of stdout.

---

## 6. Deviations from the cleanup plan, and open items

- **`latest_run.json` was not overwritten.** The plan assumed scenes 6-10 needed
  reconstructing into it. They were already there: the file already held all 10 scenes plus
  aggregates and bootstrap CIs. Writing a bare five-scene list over it would have destroyed
  the scenes 1-5 results and every aggregate. Recovery output went to
  `pipeline/eval/results/recover_scenes_6_10.json` instead, and `latest_run.json` is
  untouched at its original 6,073 bytes.
- **The cache paths in the cleanup list belonged to scenes 1-5, not 6-10.** The listed visit
  ids 416407 / 416411 / 416418 / 421069 / 467138 map to 41418155 / 41418140 / 41418135 /
  42444474 / 47333462. Scenes 6-10 are visits 421065 / 421063 / 421061 / 421060 / 421062.
  Moot in practice — `cache/arkitscenes/` and `samples/arkitscenes/raw/` were already empty,
  so all ten listed paths were absent and nothing was removed by those two commands.
- **No leaked file handles.** `lsof +L1` returned no deleted-but-open files for
  `arkitscenes` or anywhere under `spatial-ai`, so the crashed session left nothing holding
  disk or memory.
- **`docs/` is git-ignored** (`.gitignore:7`), so this report is not trackable at its
  requested path. It is written where asked; move it outside `docs/` or add a negation to
  `.gitignore` if it should be committed.
- **Open item:** commit the evaluation harness and results. Until `pipeline/eval/` is in
  git, the N=10 result set is one `rm` away from the same fate as the laser scans.
- **Open item:** re-downloading laser `.ply` files for the 10 scenes is the only way to
  restore independent ground-truth verification. Deliberately not done here.
