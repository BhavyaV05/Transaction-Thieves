"""data: Load AMLworld CSVs to typed Parquet, parse the pattern file, temporal split, 30 bank views, aliasing, held-out-cycles switch.

Owner: T1 (Abhishek).  Component: Pipeline (plan 5).  Tier: Core.
Spec: plan.md 5, tracks.md 3.3.

First steps:
  - Parquet conversion with stable tx_id
  - pattern file -> (attempt_id, typology)
  - split days 1-9 / 10-12 / 13-16; the test loader must call ftm.guards.require_final_run()
  - bank views for the 30 largest banks
  - alias mapping (server side only)
  - document schemas in ftm/data/README.md by end of week 1 (hand-off #1)
"""
