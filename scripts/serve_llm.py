"""Start the shared vLLM server (OpenAI-compatible) for one model profile. See docs/hardware.md section 5.

Owner: T3 (Akshat).  Spec: tracks.md 1.2 rules 2-3 and 5.3.  Profiles: configs/llm/serving.yaml.

    python scripts/serve_llm.py --gen-key                    # once: writes FTM_LLM_API_KEY to .env (gitignored)
    python scripts/serve_llm.py --model qwen3-8b --dry-run   # print the command, run nothing
    python scripts/serve_llm.py --model qwen3-8b             # Windows: Docker Desktop; Linux/WSL: --backend native

Safety: the API key is passed through the environment (VLLM_API_KEY), never on the command line, and the server
binds only to loopback or a private/tailnet address (never 0.0.0.0 or a public IP).
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import platform
import secrets
import shlex
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "llm" / "serving.yaml"
ENV_FILE = ROOT / ".env"
KEY_VAR = "FTM_LLM_API_KEY"
TAILNET = ipaddress.ip_network("100.64.0.0/10")  # Tailscale CGNAT range


def load_config(path: Path = CONFIG) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def check_host(host: str) -> str:
    """Allow loopback, private and tailnet addresses only (tracks.md 1.2 rule 2: never the public internet)."""
    if host == "localhost":
        return host
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as e:
        raise SystemExit(f"host must be an IP address or 'localhost', got {host!r}") from e
    if ip.is_unspecified:
        raise SystemExit(f"refusing to bind {host}: wildcard addresses expose the server on every interface")
    if not (ip.is_loopback or ip.is_private or ip in TAILNET):
        raise SystemExit(f"refusing to bind public address {host}")
    return host


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def api_key(env: dict[str, str] | None = None, env_file: Path = ENV_FILE) -> str:
    env = os.environ if env is None else env
    key = env.get(KEY_VAR) or read_env_file(env_file).get(KEY_VAR)
    if not key or key == "none":
        raise SystemExit(f"no API key: set {KEY_VAR} or run `python scripts/serve_llm.py --gen-key`")
    return key


def gen_key(env_file: Path = ENV_FILE) -> bool:
    """Append a fresh key to .env unless one is there. Returns True if a key was written."""
    if read_env_file(env_file).get(KEY_VAR):
        return False
    with env_file.open("a") as f:
        f.write(f"{KEY_VAR}={secrets.token_urlsafe(32)}\n")
    return True


def vllm_args(cfg: dict, name: str, port: int, host: str | None = None) -> list[str]:
    """Arguments after `vllm serve` / the image entrypoint. Identical on every machine for a given profile."""
    if name not in cfg["models"]:
        raise SystemExit(f"unknown model profile {name!r}; choose from {sorted(cfg['models'])}")
    m, common = cfg["models"][name], cfg.get("common", {})
    args = [
        "--model", m["repo"],
        "--revision", m["revision"],
        "--served-model-name", name,
        "--max-model-len", str(m["max_model_len"]),
        "--gpu-memory-utilization", str(m["gpu_memory_utilization"]),
        "--seed", str(common.get("seed", 0)),
        "--port", str(port),
    ]  # fmt: skip
    if host is not None:
        args += ["--host", host]
    if common.get("generation_config"):
        args += ["--generation-config", common["generation_config"]]
    if common.get("enable_prefix_caching"):
        args.append("--enable-prefix-caching")
    if m.get("reasoning_parser"):
        args += ["--reasoning-parser", m["reasoning_parser"]]
    return args


def docker_command(cfg: dict, name: str, host: str, port: int, hf_home: str) -> list[str]:
    """Docker Desktop (Windows) or Docker on Linux. `-e VLLM_API_KEY` copies the key from our environment."""
    return [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "--name", f"ftm-llm-{name}",
        "-p", f"{host}:{port}:{port}",
        "-v", f"{hf_home}:/root/.cache/huggingface",
        "-e", "VLLM_API_KEY", "-e", "HF_TOKEN",
        cfg["docker_image"],
        *vllm_args(cfg, name, port, host="0.0.0.0"),  # inside the container; the -p mapping restricts the host side
    ]  # fmt: skip


def native_command(cfg: dict, name: str, host: str, port: int) -> list[str]:
    """Linux or WSL2 with `pip install vllm==<vllm_version>`."""
    return ["vllm", "serve", *vllm_args(cfg, name, port, host=host)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-8b", help="profile in configs/llm/serving.yaml")
    ap.add_argument("--backend", choices=["docker", "native"], default=None,
                    help="default: docker on Windows, native elsewhere")  # fmt: skip
    ap.add_argument("--host", default=None, help="bind address (default from config: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    ap.add_argument("--gen-key", action="store_true", help=f"write a new {KEY_VAR} to .env if absent, then exit")
    a = ap.parse_args(argv)

    if a.gen_key:
        print(f"wrote {KEY_VAR} to {ENV_FILE}" if gen_key() else f"{KEY_VAR} already in {ENV_FILE}; kept it")
        return 0

    cfg = load_config(a.config)
    host = check_host(a.host or cfg["host"])
    port = a.port or cfg["port"]
    backend = a.backend or ("docker" if platform.system() == "Windows" else "native")
    hf_home = str(Path(os.path.expanduser(cfg["hf_home"])))
    if backend == "docker":
        cmd = docker_command(cfg, a.model, host, port, hf_home)
    else:
        cmd = native_command(cfg, a.model, host, port)

    if platform.system() == "Windows" and Path(hf_home).drive.upper() == "C:":
        print(f"WARNING: the HF cache {hf_home} is on C:. Weights are 5-13 GB each; set FTM_HF_HOME (e.g. E:\\hf_cache)")
    print(f"# vLLM {cfg['vllm_version']}, profile {a.model}, HF cache {hf_home}, endpoint http://{host}:{port}/v1")
    print(shlex.join(cmd))
    if a.dry_run:
        return 0
    env = dict(os.environ, VLLM_API_KEY=api_key())
    if backend == "native":
        env["HF_HOME"] = hf_home
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
