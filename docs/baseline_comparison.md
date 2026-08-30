# Baseline Comparison Notes

## RTAB-Map (3D SLAM) Failure

During our live execution on the VM for the RTAB-Map baseline, the system suffered from a catastrophic scale collapse and trajectory failure. Specifically, the visual odometry logs showed:
- `inliers=0/0` transitioning to `30/30`
- `3.29cm` estimated motion

These figures are recorded here as literal quoted text from the VM session, as the raw logs are no longer available after the VM teardown. Because of this tracking failure, RTAB-Map could not produce a valid reconstructed room cloud or height for this scene.

## Scene 42444733 Geometry Failure

For scene `42444733` (visit `421267`), our spatial AI pipeline's geometry run failed midway and **never produced a height prediction**. 

Therefore, the three-way comparison (FARO Laser Scanner vs. RTAB-Map vs. VIO-script) for this specific scene did **not** actually include our own pipeline's result. This gap is explicitly noted to ensure the comparison table accurately reflects that our deterministic 3D pipeline yielded no output for scene `42444733`. The laser ground-truth height for this scene was manually recorded as `2.3442m` for reference.
