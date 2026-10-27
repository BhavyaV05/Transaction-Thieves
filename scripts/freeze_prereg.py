"""Freeze the pre-registration: write prereg/FREEZE.json and print the combined hash to send to the TA.

Owner: T4 (Shreyas).  Spec: tracks.md 8, prereg/README.md, docs/t4_final_run.md (the 30 Oct checklist).  Run from the repo root.

    # 1. prereg/ committed and clean: write the manifest (sha256 of every prereg/*.yaml and subset_manifest.txt, the commit)
    python scripts/freeze_prereg.py
    # 2. send the printed combined hash to the TA; when it has gone out, record that
    python scripts/freeze_prereg.py --record-ta-sent <combined hash>
    # 3. commit prereg/FREEZE.json
    # check only (no write): do the files still match?
    python scripts/freeze_prereg.py --check

This is the only writer of FREEZE.json. scripts/run_final.py refuses to run unless the recomputed hashes match it, the
hash was recorded as sent, the tree is clean and the recorded commit is in HEAD's history. subset_manifest.txt is hashed
without its `subset_sha256:` line, which is filled in after the final run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ftm.final import freeze as fz  # noqa: E402
from ftm.final import preflight as pf  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-ta-sent", metavar="HASH", default=None, help="record the hash that was sent to the TA")
    ap.add_argument("--check", action="store_true", help="verify the files against FREEZE.json; write nothing")
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--prereg-dir", type=Path, default=ROOT / "prereg")
    a = ap.parse_args(argv)
    paths = pf.Paths(repo=a.repo, prereg=a.prereg_dir)

    if a.check:
        problems = pf.check_freeze_manifest(paths) or pf.check_hashes_match(paths)
        problems += pf.check_commit_ancestor(paths) if not problems else []
        print("\n".join(problems) if problems else f"ok: {fz.read_freeze(a.prereg_dir)['combined_sha256']}")
        return 1 if problems else 0
    if a.record_ta_sent:
        fr = fz.record_ta_hash(a.prereg_dir, a.record_ta_sent)
        print(f"recorded: the TA was sent {fr['ta_ack_hash']}. Commit {a.prereg_dir.name}/{fz.FREEZE_FILE}.")
        return 0
    fr = fz.write_freeze(a.repo, a.prereg_dir)
    unresolved = pf.unresolved_fields(a.prereg_dir)
    print(f"wrote {a.prereg_dir.name}/{fz.FREEZE_FILE} at commit {fr['git_commit'][:10]}")
    for name, h in fr["files"].items():
        print(f"  {h}  {name}")
    print(f"\nCOMBINED HASH (send this to the TA):\n{fr['combined_sha256']}")
    if unresolved:
        print(f"\nWARNING: {len(unresolved)} values are still fix_on_val: " + ", ".join(unresolved))
    print("\nnext: send the hash, then  python scripts/freeze_prereg.py --record-ta-sent <hash>  and commit FREEZE.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
