"""Self-driving task runner for one ada allocation (4 GPUs).  Owner: T3 (Akshat).

Reads ~/tt/tasks.txt, one task per line:  name|n_gpus|n_cpus|mem_gb|command   (priority order; '#' lines ignored).  Runs tasks in parallel as long as free GPUs,
CPUs (24) and RAM (100 GB) allow, each with its own CUDA_VISIBLE_DEVICES; a task whose resources do not fit yet is skipped for now (later ones may start).
A task with a done marker (~/tt/results/done/<name>) is skipped; a failed task is retried once; per-task log ~/tt/results/logs/<name>.log; timings in
~/tt/results/walltimes.tsv.  Exits when every task is done or has failed twice, so the allocation is released.
"""

import os
import subprocess
import time
from pathlib import Path

HOME = Path.home()
RES = HOME / "tt" / "results"
(RES / "logs").mkdir(parents=True, exist_ok=True)
(RES / "done").mkdir(exist_ok=True)
CPUS, MEM, GPUS = 24, 100, [0, 1, 2, 3]
TASK_TIMEOUT_S = 20 * 3600


def load():
    out = []
    for line in (HOME / "tt" / "tasks.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, g, c, m, cmd = line.split("|", 4)
            out.append({"name": name, "g": int(g), "c": int(c), "m": int(m), "cmd": cmd, "tries": 0})
    return out


def main():
    tasks = [t for t in load() if not (RES / "done" / t["name"]).exists()]
    running, free = [], list(GPUS)
    print(time.strftime("%F %T"), "tasks:", [t["name"] for t in tasks], flush=True)
    while tasks or running:
        for r in list(running):
            rc = r["p"].poll()
            timed_out = rc is None and time.time() - r["t0"] > TASK_TIMEOUT_S
            if timed_out:
                r["p"].kill()
                rc = -9
            if rc is None:
                continue
            running.remove(r)
            free += r["gpus"]
            wall = time.time() - r["t0"]
            with open(RES / "walltimes.tsv", "a") as f:
                f.write(f"{r['t']['name']}\t{time.strftime('%F %T')}\trc={rc}\twall_s={wall:.0f}\ttry={r['t']['tries']}\tgpus={r['gpus']}\n")
            print(time.strftime("%F %T"), r["t"]["name"], "rc", rc, f"{wall:.0f}s", flush=True)
            if rc == 0:
                (RES / "done" / r["t"]["name"]).write_text(f"{wall:.0f}s\n")
            elif r["t"]["tries"] < 2:
                tasks.insert(0, r["t"])  # retry once, first in line
        used_c, used_m = sum(r["t"]["c"] for r in running), sum(r["t"]["m"] for r in running)
        for t in list(tasks):
            if t["g"] <= len(free) and used_c + t["c"] <= CPUS and used_m + t["m"] <= MEM:
                gpus, free = free[: t["g"]], free[t["g"]:]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), OMP_NUM_THREADS=str(max(1, t["c"])))
                t["tries"] += 1
                log = open(RES / "logs" / f"{t['name']}.log", "a")
                log.write(f"\n=== {time.strftime('%F %T')} try {t['tries']} gpus {gpus}: {t['cmd']}\n")
                log.flush()
                p = subprocess.Popen(["bash", "-c", t["cmd"]], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running.append({"t": t, "p": p, "gpus": gpus, "t0": time.time()})
                tasks.remove(t)
                used_c, used_m = used_c + t["c"], used_m + t["m"]
                print(time.strftime("%F %T"), "start", t["name"], "gpus", gpus, flush=True)
        time.sleep(10)
    print(time.strftime("%F %T"), "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
