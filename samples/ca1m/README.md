# Apple CA-1M held-out validation samples

Raw archives live under `samples/ca1m/raw/` and are ignored by Git.

## Why this dataset

CA-1M exists here to fix a specific weakness. The ARKitScenes FARO comparison (V02)
could not establish a correspondence between the ARKit session frame and the FARO visit
frame, so only gravity-invariant height was comparable and even that was measured
against a reference pairing that was itself ambiguous. CA-1M publishes, per frame, a
camera pose **already registered into the laser scanner's frame** plus a depth image
rendered from that laser scan. That is the correspondence V02 lacked.

## License

Apple releases CA-1M under
[CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/): non-commercial,
no derivatives. Raw archives stay local and are never committed or redistributed. Only
derived measurements and hashes are recorded in this repository.

## What CA-1M does and does not publish

Per frame, inside `ca1m-val-<id>.tar`:

| path | role |
| --- | --- |
| `<ts>.wide/image.png` | 1024x768 RGB — mobile input |
| `<ts>.wide/depth.png` | 256x192 uint16 mm ARKit depth — mobile input |
| `<ts>.wide/image/K.json`, `<ts>.wide/depth/K.json` | intrinsics — mobile input |
| `<ts>.wide/T_gravity.json` | 3x3 pitch/roll — mobile input |
| `<ts>.gt/RT.json` | 4x4 pose registered in laser space — **evaluator only** |
| `<ts>.gt/depth.png` | 512x384 uint16 mm FARO-rendered depth — **evaluator only** |
| `<ts>.wide/instances.json`, `world.gt/instances.json` | 3D boxes — **evaluator only** |

**CA-1M publishes no mobile camera pose.** `cubifyanything/dataset.py` sets the wide
sensor's `RT` to the identity and keeps mobile data in camera coordinates; the only pose
released is the laser-registered one, which is ground truth. A reconstruction fed that
pose would be handed the answer.

The mobile trajectory therefore comes from **ARKitScenes Validation**
(`lowres_wide.traj`), which is the same capture's on-device ARKit odometry. CA-1M and
ARKitScenes share the underlying captures and, as verified on `45261179`, the same
timestamp base, so frames pair directly.

```text
reconstruction sees   ARKitScenes RGB, ARKit depth, intrinsics, ARKit odometry
evaluator sees        CA-1M gt/RT.json and gt/depth.png, after the prediction is hashed
```

`pipeline/tests/test_ca1m_leakage.py` enforces that split: no inference module may name
a ground-truth asset or import the evaluator, the prediction hash is taken before any
`gt/` file is opened, and `solve_alignment` is given poses only, never the model.

## Held-out status

These are ARKitScenes **Validation**-split captures. The frozen geometry configuration
was fixed against Training scene `47333462`. No capture here contributed to any
parameter, and none may drive a threshold change.

The selection is frozen in
[`validation/manifests/ca1m_heldout_manifest.json`](../../validation/manifests/ca1m_heldout_manifest.json)
before any error was computed. The rule is Apple's own `data/val.txt` order, first five,
excluding the capture used for the one-sample acceptance gate.

## Known limit on what these numbers can mean

Alignment between the reconstruction's ARKit frame and laser space is solved from paired
poses. ARKit odometry drifts while the reference poses are registered per frame, so that
residual **is** the capture's drift. On the gate capture it was 1.46 deg and 5 cm at p90,
which smears horizontal bands by roughly 17 cm across the room. Global surface distances
at or below that scale are not separable from drift. The reported storey height avoids
this by fitting a plane to each band and measuring between the fitted planes, which is
invariant under a rigid rotation.
