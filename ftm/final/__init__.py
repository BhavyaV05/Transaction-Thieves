"""final: the G5 final run (freeze manifest, preflight, job plan, resumable job runner).

Owner: T4 (Shreyas).  Spec: plan.md 8 and 13 (G5), tracks.md 6.7 and 8.  Entry point: scripts/run_final.py.
Nothing in this package sets FTM_FINAL_RUN; only scripts/run_final.py does, inside main(), after the preflight.
"""
