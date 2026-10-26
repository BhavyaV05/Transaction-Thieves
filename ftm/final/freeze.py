"""Pre-registration freeze manifest: hashes of prereg/*.yaml and prereg/subset_manifest.txt, the git commit, the TA hash.

Owner: T4 (Shreyas).  Spec: tracks.md 8 ("after the pre-registration hash has been sent"), prereg/README.md.

`prereg/FREEZE.json` is written only by scripts/freeze_prereg.py. scripts/run_final.py reads it in its preflight.
The combined hash is the sha256 of the sorted lines "<file name> <sha256>"; it is what is sent to the TA.
`subset_manifest.txt` is hashed without its `subset_sha256:` line: that line is filled in after the final run
(the run prints it), and filling it in must not break the freeze.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

FREEZE_FILE = "FREEZE.json"
SUBSET_MANIFEST = "subset_manifest.txt"
FREEZE_VERSION = 1
_SUBSET_HASH_LINE = re.compile(r"^\s*#?\s*subset_sha256\s*:.*$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_bytes(name: str, data: bytes) -> bytes:
    """The bytes that are hashed: the file itself, except that subset_manifest.txt loses its `subset_sha256:` line."""
    if name == SUBSET_MANIFEST:
        text = data.decode("utf-8")
        kept = [line for line in text.splitlines() if not _SUBSET_HASH_LINE.match(line)]
        return ("\n".join(kept) + "\n").encode("utf-8")
    return data


def frozen_names(prereg_dir: Path) -> list[str]:
    names = sorted(p.name for p in Path(prereg_dir).glob("*.yaml"))
    if (Path(prereg_dir) / SUBSET_MANIFEST).exists():
        names.append(SUBSET_MANIFEST)
    return names


def hash_files(prereg_dir: Path) -> dict[str, str]:
    d = Path(prereg_dir)
    return {n: sha256_bytes(normalized_bytes(n, (d / n).read_bytes())) for n in frozen_names(d)}


def combined_hash(hashes: dict[str, str]) -> str:
    return sha256_bytes("".join(f"{n} {hashes[n]}\n" for n in sorted(hashes)).encode("utf-8"))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT).strip()


def git_ok(repo: Path, *args: str) -> bool:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True).returncode == 0


def tree_is_clean(repo: Path) -> bool:
    """No tracked change and no untracked file (ignored files such as outputs/ do not count)."""
    return git(repo, "status", "--porcelain") == ""


def read_freeze(prereg_dir: Path) -> dict:
    return json.loads((Path(prereg_dir) / FREEZE_FILE).read_text(encoding="utf-8"))


def build_freeze(repo: Path, prereg_dir: Path, ta_ack_hash: str | None = None) -> dict:
    hashes = hash_files(prereg_dir)
    return {"version": FREEZE_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": git(repo, "rev-parse", "HEAD"), "files": hashes,
            "combined_sha256": combined_hash(hashes), "ta_ack_hash": ta_ack_hash}  # fmt: skip


def write_freeze(repo: Path, prereg_dir: Path, ta_ack_hash: str | None = None) -> dict:
    """Create prereg/FREEZE.json for the current commit. Refuses a dirty tree (the commit must hold the frozen files);
    the only exception is FREEZE.json itself, which is what is being (re)written."""
    me = (Path(prereg_dir).resolve().relative_to(Path(repo).resolve()) / FREEZE_FILE).as_posix()
    dirty = [ln for ln in git(repo, "status", "--porcelain").splitlines() if ln[3:].strip() != me]
    if dirty:
        raise RuntimeError("working tree is not clean; commit the prereg files first:\n" + "\n".join(dirty))
    fr = build_freeze(repo, prereg_dir, ta_ack_hash)
    (Path(prereg_dir) / FREEZE_FILE).write_text(json.dumps(fr, indent=2) + "\n", encoding="utf-8")
    return fr


def record_ta_hash(prereg_dir: Path, sent_hash: str) -> dict:
    """Record the hash that was sent to the TA. It must equal the current combined hash of the files."""
    fr = read_freeze(prereg_dir)
    now = combined_hash(hash_files(prereg_dir))
    if sent_hash != now or sent_hash != fr["combined_sha256"]:
        raise RuntimeError(f"hash sent ({sent_hash}) does not match the files ({now}) / FREEZE.json ({fr['combined_sha256']}); "
                           "re-run freeze_prereg.py and send the new hash")
    fr["ta_ack_hash"] = sent_hash
    fr["ta_ack_recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (Path(prereg_dir) / FREEZE_FILE).write_text(json.dumps(fr, indent=2) + "\n", encoding="utf-8")
    return fr
