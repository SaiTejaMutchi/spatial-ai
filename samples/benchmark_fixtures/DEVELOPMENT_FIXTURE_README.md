# DEVELOPMENT BENCHMARK FIXTURES — NOT REAL MEASUREMENTS

Every value in this directory is **invented** to exercise the benchmark
software: the parser, the error formulas, the missing-value paths, and the
report writer. Nothing here was measured from any room.

These files must never appear in a final benchmark, a report, or any accuracy
claim. The final tape-measure ground truth is supplied by a human after
`DEV_COMPLETE` and arrives through this same parser unchanged.

`measurement_id` values deliberately do **not** match the IDs any real scan
produces, so a fixture row cannot silently join a real comparison.
