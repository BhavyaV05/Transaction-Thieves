"""T3 week 0: serve_llm.py command building and safety, bench_llm.py maths, the stdlib chat client."""

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ftm.llm.client import ChatResponse, HttpChatClient, LLMError, parse_completion

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


serve = _load("serve_llm")
bench = _load("bench_llm")


@pytest.fixture(scope="module")
def cfg():
    return serve.load_config()


def test_every_profile_is_pinned(cfg):
    assert cfg["docker_image"].endswith(f"v{cfg['vllm_version']}")
    for name, m in cfg["models"].items():
        assert len(m["revision"]) == 40, name  # a full commit hash, not a branch
        assert m["max_model_len"] >= 6000 + 512, name  # fits Limits.max_tokens plus an answer


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "100.101.102.103", "192.168.1.20"])
def test_private_hosts_allowed(host):
    assert serve.check_host(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "8.8.8.8", "example.com"])
def test_wildcard_and_public_hosts_refused(host):
    with pytest.raises(SystemExit):
        serve.check_host(host)


def test_docker_command_hides_key_and_binds_host_side(cfg):
    cmd = serve.docker_command(cfg, "qwen3-8b", "127.0.0.1", 8000, "E:/hf_cache")
    joined = " ".join(cmd)
    assert "-p 127.0.0.1:8000:8000" in joined
    assert "E:/hf_cache:/root/.cache/huggingface" in joined
    assert "-e VLLM_API_KEY" in joined and "--api-key" not in joined
    assert cmd[cmd.index("--revision") + 1] == cfg["models"]["qwen3-8b"]["revision"]
    assert "--reasoning-parser" in cmd


def test_native_and_docker_share_server_flags(cfg):
    d = serve.docker_command(cfg, "gpt-oss-20b", "127.0.0.1", 8000, "/hf")
    n = serve.native_command(cfg, "gpt-oss-20b", "127.0.0.1", 8000)

    def flags(c):
        return [x for x in c[c.index("--model"):] if x not in {"0.0.0.0", "127.0.0.1"}]

    assert flags(d) == flags(n)
    assert "--reasoning-parser" not in n


def test_unknown_profile(cfg):
    with pytest.raises(SystemExit):
        serve.vllm_args(cfg, "nope", 8000)


def test_gen_key_writes_once(tmp_path):
    env = tmp_path / ".env"
    assert serve.gen_key(env) is True
    first = env.read_text()
    assert serve.gen_key(env) is False and env.read_text() == first
    assert serve.api_key(env={}, env_file=env) == first.split("=", 1)[1].strip()
    with pytest.raises(SystemExit):
        serve.api_key(env={}, env_file=tmp_path / "missing")


def test_bench_prompt_is_fixed_but_cache_busting():
    a, b = bench.bench_prompt(0, "a"), bench.bench_prompt(0, "a")
    assert a == b
    c = bench.bench_prompt(1, "a")
    assert a[1]["content"] != c[1]["content"]
    assert a[1]["content"].split("\n", 1)[1] == c[1]["content"].split("\n", 1)[1]  # only line 1 differs


class FakeClient:
    """Prefill costs 0.001 s per prompt token, decode 0.02 s per token."""

    def chat(self, messages, *, max_tokens, **kw):
        n_in = len(messages[1]["content"]) // 3
        lat = 0.001 * n_in + 0.02 * (max_tokens - 1)
        return ChatResponse(text="x", tokens_in=n_in, tokens_out=max_tokens, latency_s=lat, model="fake")


def test_bench_maths_recovers_known_speeds():
    runs = [bench.one_run(FakeClient(), r, max_tokens=101) for r in range(3)]
    s = bench.summarise(runs, None, 101)
    assert s["median_decode_tok_s"] == pytest.approx(50.0, rel=0.02)
    assert s["median_prefill_tok_s"] == pytest.approx(1000.0, rel=0.02)
    assert s["short_outputs"] == 0
    assert bench.factor({"median_decode_tok_s": 25.0}, s) == pytest.approx(0.5, rel=0.02)


def test_parse_completion_and_errors():
    body = {
        "model": "qwen3-8b",
        "choices": [{"message": {"content": "{}", "reasoning_content": "hmm"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
    r = parse_completion(body, 0.5)
    assert (r.text, r.reasoning, r.tokens_in, r.tokens_out) == ("{}", "hmm", 12, 3)
    with pytest.raises(LLMError):
        parse_completion({"error": "boom"}, 0.1)


def test_client_body_has_schema_and_extra():
    c = HttpChatClient("http://x/v1", api_key="none", model="m",
                       default_extra={"chat_template_kwargs": {"enable_thinking": False}})  # fmt: skip
    assert c.api_key is None
    body = c.build_body([{"role": "user", "content": "hi"}], json_schema={"title": "step", "type": "object"},
                        extra={"ignore_eos": True})  # fmt: skip
    assert body["response_format"]["json_schema"]["name"] == "step"
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["ignore_eos"] is True


# --- llama.cpp backend (GGUF candidates of the E18 sweep) ---

LLAMA = {"served_model": "g", "max_model_len": 8192, "gguf_repo": "o/r-GGUF", "gguf_file": "m-Q3.gguf", "stack": "llama.cpp"}


def test_llamacpp_command_construction():
    cmd = serve.llamacpp_command(LLAMA, "g", "/hf/m-Q3.gguf", "127.0.0.1", 8000, seed=0)
    assert cmd[1:3] == ["-m", "/hf/m-Q3.gguf"]
    for flag, val in [("--alias", "g"), ("--ctx-size", "8192"), ("--n-gpu-layers", "all"), ("--parallel", "1"),
                      ("--seed", "0"), ("--host", "127.0.0.1"), ("--port", "8000")]:  # fmt: skip
        assert cmd[cmd.index(flag) + 1] == val
    assert "--jinja" in cmd
    assert cmd[cmd.index("--chat-template-kwargs") + 1] == '{"enable_thinking":false}'
    two = serve.llamacpp_command({**LLAMA, "parallel": 2}, "g", "/x", "127.0.0.1", 8000)
    assert two[two.index("--ctx-size") + 1] == "16384"  # ctx is shared across slots


def test_llamacpp_key_never_in_argv():
    cmd = serve.llamacpp_command(LLAMA, "g", "/x", "127.0.0.1", 8000)
    assert not any("api-key" in c.lower() for c in cmd)


def test_e18_gguf_profiles_select_llamacpp_and_stack_recorded():
    from ftm.planners import e18  # run_e18 records cand.stack as the serving stack

    cands = {c.name: c for c in e18.candidates(e18.load_config(serve.SWEEP_CONFIG), ("core", "if_time"))}
    for name in ("qwen3.8-27b-q3", "gemma4-31b-q3"):
        p = serve.llama_profile(serve.SWEEP_CONFIG, name)
        assert p["stack"] == "llama.cpp" and p["gguf_repo"] and p["gguf_file"] and p["max_model_len"] == 8192
        assert cands[name].stack == "llama.cpp"
    assert serve.llama_profile(serve.SWEEP_CONFIG, "nope") is None


def test_find_gguf_missing_gives_download_command(tmp_path):
    with pytest.raises(SystemExit) as e:
        serve.find_gguf(str(tmp_path), "o/r-GGUF", "m-Q3.gguf", "abc")
    assert "hf download o/r-GGUF --include m-Q3.gguf --revision abc" in str(e.value)


def test_find_gguf_resolves_cache(tmp_path):
    f = tmp_path / "hub" / "models--o--r-GGUF" / "snapshots" / "abc" / "m-Q3.gguf"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"")
    assert serve.find_gguf(str(tmp_path), "o/r-GGUF", "m-Q3.gguf", "abc") == f.resolve()
    assert serve.find_gguf(str(tmp_path), "o/r-GGUF", "m-Q3.gguf") == f.resolve()  # unpinned revision


def test_llamacpp_dry_run_hides_key_and_checks_host(tmp_path, monkeypatch, capsys):
    rev = OmegaConf.load(ROOT / "configs" / "experiments" / "e18_backbone_sweep.yaml").serving["qwen3.8-27b-q3"].revision
    f = tmp_path / "hub" / "models--unsloth--Qwen3.8-27B-GGUF" / "snapshots" / str(rev) / "Qwen3.8-27B-UD-Q3_K_XL.gguf"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"")
    monkeypatch.setenv("FTM_HF_HOME", str(tmp_path))
    monkeypatch.setenv("FTM_LLM_API_KEY", "sekret-key")
    assert serve.main(["--model", "qwen3.8-27b-q3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "llama-server" in out and "--jinja" in out and "sekret-key" not in out and "--api-key" not in out
    with pytest.raises(SystemExit):
        serve.main(["--model", "qwen3.8-27b-q3", "--host", "0.0.0.0", "--dry-run"])
