"""Pre-registration freeze manifest: hashes of prereg/*.yaml, prereg/subset_manifest.txt and EXTRA_FROZEN, the git commit, the TA hash.

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
# Files outside prereg/ that the hash also covers (repo-relative, always with a "/", so a name never clashes with a prereg file):
# T2's non-LLM arm settings and the question tables they read (asked by T2 in PR #76, 28 Oct). Missing files are skipped here and
# reported by scripts/freeze_prereg.py.
EXTRA_FROZEN = (
    "configs/final/non_llm.yaml", "configs/experiments/ablations_nonllm.yaml",
    "configs/final/top_external.json",  # the G4 pick, chosen on validation
    # the pieces fitted on train / val fit+cal before test existed (headroom_v1, docs/t3_headroom.md) and the p0 map: a change after the freeze changes the hash
    "configs/llm/tuned.yaml", "configs/fusion/headroom_v1.json", "configs/fusion/headroom_v1_table.jsonl", "configs/fusion/headroom_v1_decisions.json",
    "configs/fusion/headroom_v1_accepted.json", "configs/fusion/headroom_v1_accepted_table.jsonl", "configs/decision/headroom_v1.yaml",
    "configs/t1_final/p0_recal_hi_medium.json",
    "configs/fusion/enumerator_table.jsonl",  # the H3 enumerator arm's table (non_llm.yaml tables.enumerator), built on train 8 Nov
)  # fmt: skip
# Hashed like EXTRA_FROZEN but not tracked by git (outputs/ is ignored): the DSPy MIPROv2 program the dspy_react jobs load
# (--dspy-program in configs/final/jobs.yaml). The preflight compares its hash with FREEZE.json instead of `git show`.
def _common_library() -> tuple[str, ...]:
    """The strategy library every LLM job loads: the `--library` of `llm.common_args` in configs/final/jobs.yaml, read from there so the freeze and the jobs cannot disagree
    (and so this file does not name a library path of its own)."""
    import yaml

    try:
        args = (yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "final" / "jobs.yaml").read_text()).get("llm") or {}).get("common_args") or []
        return (str(args[args.index("--library") + 1]),) if "--library" in args else ()
    except (OSError, ValueError, IndexError, AttributeError):
        return ()


# [0] must stay the DSPy program (preflight.check_dspy_program); the frozen strategy library of the jobs' common args follows.
EXTRA_UNTRACKED = ("outputs/t3_dspy/react_miprov2.json", *_common_library())
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


def frozen_path(prereg_dir: Path, name: str) -> Path:
    """prereg/<name>, or <repo>/<name> for a name with a "/" (EXTRA_FROZEN); the repo is the parent of prereg/."""
    d = Path(prereg_dir)
    return d.parent / name if "/" in name else d / name


def missing_extra(prereg_dir: Path) -> list[str]:
    return [n for n in EXTRA_FROZEN if not frozen_path(prereg_dir, n).exists()]


def missing_untracked(prereg_dir: Path) -> list[str]:
    return [n for n in EXTRA_UNTRACKED if not frozen_path(prereg_dir, n).exists()]


def frozen_names(prereg_dir: Path) -> list[str]:
    names = sorted(p.name for p in Path(prereg_dir).glob("*.yaml"))
    if (Path(prereg_dir) / SUBSET_MANIFEST).exists():
        names.append(SUBSET_MANIFEST)
    return names + [n for n in (*EXTRA_FROZEN, *EXTRA_UNTRACKED) if frozen_path(prereg_dir, n).exists()]


def hash_files(prereg_dir: Path) -> dict[str, str]:
    d = Path(prereg_dir)
    return {n: sha256_bytes(normalized_bytes(n, frozen_path(d, n).read_bytes())) for n in frozen_names(d)}


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
    dirty = [ln for ln in git(repo, "status", "--porcelain").splitlines() if ln.split(None, 1)[-1].strip() != me]  # git() strips the first line's leading space
    if dirty:
        raise RuntimeError("working tree is not clean; commit the prereg files first:\n" + "\n".join(dirty))
    fr = build_freeze(repo, prereg_dir, ta_ack_hash)
    (Path(prereg_dir) / FREEZE_FILE).write_text(json.dumps(fr, indent=2) + "\n", encoding="utf-8")
    return fr


def subset_sha_line(prereg_dir: Path) -> str | None:
    """The current `subset_sha256:` line of subset_manifest.txt, or None if there is none."""
    f = Path(prereg_dir) / SUBSET_MANIFEST
    for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
        if _SUBSET_HASH_LINE.match(line):
            return line
    return None


def write_subset_sha(prereg_dir: Path, sha: str, note: str) -> bool:
    """Fill the `# subset_sha256:` line of prereg/subset_manifest.txt (T4's file) with the sha256 of the sorted subset ids. That line
    is not part of the freeze hash, and no other line is touched (rule_commit stays as it is). A placeholder is replaced; the same
    hash again is a no-op (returns False); a different hash already recorded raises, because the subset is frozen once."""
    f = Path(prereg_dir) / SUBSET_MANIFEST
    lines = f.read_text(encoding="utf-8").splitlines()
    new = f"# subset_sha256: {sha}  # {note}"
    for i, line in enumerate(lines):
        if _SUBSET_HASH_LINE.match(line):
            have = line.split(":", 1)[1].strip().split()[0] if line.split(":", 1)[1].strip() else ""
            if re.fullmatch(r"[0-9a-f]{64}", have):
                if have == sha:
                    return False
                raise RuntimeError(f"{f.name} already records subset_sha256 {have}; this run drew {sha}. The subset is frozen once")
            lines[i] = new
            break
    else:
        lines.append(new)
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def record_ta_waiver(prereg_dir: Path, hash_: str) -> dict:
    """Record that the hash is NOT sent to any TA (6 Nov 2026 decision): the freeze then rests on the git commit of FREEZE.json alone.

    The report must say so; the hash must equal the current combined hash, as for record_ta_hash."""
    fr = read_freeze(prereg_dir)
    now = combined_hash(hash_files(prereg_dir))
    if hash_ != now or hash_ != fr["combined_sha256"]:
        raise RuntimeError(f"hash given ({hash_}) does not match the files ({now}) / FREEZE.json ({fr['combined_sha256']}); re-run freeze_prereg.py")
    fr["ta_waived_hash"] = hash_
    fr["ta_waived_recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    fr["ta_waived_note"] = "not sent to any TA; the pre-registration is fixed by the git commit that holds this file"
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
