"""probe.py without a GPU: python -m pytest tests

Parameters from the environment, the eval.toml contract, patching a real train_gpt.py, reading the step and
validation lines, and main() end to end for both kinds with a fake torchrun (as speedrun-perf's tests).

fixtures/train_gpt_mtpmed.py is a real train_gpt.py of the speedrun subject (modded-nanogpt-jumps, the
mtp-medium candidate of nanogpt-jumps-stacked), used only as text: the patch must find its Hyperparameters.
"""

from __future__ import annotations

import ast
import json
import math
import os
import stat
import subprocess
import sys
import textwrap
import time
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))

import bench  # noqa: E402
import probe  # noqa: E402

REAL = (HERE / "fixtures" / "train_gpt_mtpmed.py").read_text()


# ---------------------------------------------------------------- eval.toml and parameters

def test_eval_toml_is_a_procs_eval_with_the_two_kinds():
    cfg = tomllib.loads((EVAL / "eval.toml").read_text())["eval"]
    assert (cfg["type"], cfg["evaluator"], cfg["entrypoint"]) == ("procs", "deterministic", "python probe.py")
    kinds = cfg["procs"]
    assert {k["class"] for k in kinds.values()} == {"experiment"}
    assert set(kinds) == {"loss-vs-parent", "full-run-1gpu"}
    job = tomllib.loads((EVAL / "job.toml").read_text())
    for name, k in kinds.items():
        assert k["description"] and 0 < k["max_wall_s"] <= 3600 and k["stages"], name
        assert job["timeout_s"] >= k["max_wall_s"]
        for out, spec in k["outputs"].items():
            assert spec["type"] in ("float", "int", "string", "list[float]", "list[str]"), (name, out)
    assert kinds["loss-vs-parent"]["params"]["steps"] == {"type": "int", "min": 100, "max": 500, "default": 300}
    assert kinds["full-run-1gpu"]["params"]["num_iterations"] == {"type": "int", "min": 1500, "max": 2600,
                                                                  "optional": True}
    assert {"parent_val_loss", "candidate_val_loss", "val_steps", "ms_per_step", "parent_ms_per_step",
            "val_loss_delta", "critique"} <= set(kinds["loss-vs-parent"]["outputs"])
    assert {"val_loss", "steps", "train_time_s", "ms_per_step", "critique"} <= set(kinds["full-run-1gpu"]["outputs"])
    # full-run-1gpu runs one code, so Kupanga runs the parent beside it; loss-vs-parent measures the parent itself
    assert (kinds["full-run-1gpu"]["baseline"], kinds["loss-vs-parent"]["baseline"]) == ("parent", "in-job")


def test_every_stage_probe_prints_is_listed():
    kinds = probe.load_kinds()
    assert kinds["loss-vs-parent"]["stages"] == ["downloading data", "rebuilding the parent", "compiling parent",
                                                 "training parent", "compiling candidate", "training candidate"]
    assert kinds["full-run-1gpu"]["stages"] == ["downloading data", "compiling candidate", "training candidate",
                                                "validating candidate"]


def test_params_default_and_parse():
    assert probe.parse_params("loss-vs-parent", {}) == {"steps": 300}
    assert probe.parse_params("loss-vs-parent", {"PROBE_STEPS": " 500 "}) == {"steps": 500}
    assert probe.parse_params("full-run-1gpu", {}) == {"num_iterations": None}
    assert probe.parse_params("full-run-1gpu", {"PROBE_NUM_ITERATIONS": "1800"}) == {"num_iterations": 1800}


@pytest.mark.parametrize("kind,env,match", [
    ("loss-vs-parent", {"PROBE_STEPS": "99"}, "outside 100–500"),
    ("loss-vs-parent", {"PROBE_STEPS": "501"}, "outside 100–500"),
    ("loss-vs-parent", {"PROBE_STEPS": "3.5"}, "not an int"),
    ("full-run-1gpu", {"PROBE_NUM_ITERATIONS": "2601"}, "outside 1500–2600"),
    ("full-run-1gpu", {"PROBE_NUM_ITERATIONS": "abc"}, "not an int"),
    ("nope", {}, "not one of this eval's probes"),
    ("", {}, "not one of this eval's probes"),
])
def test_bad_params_are_the_evals_problem(kind, env, match):
    with pytest.raises(probe.InfraError, match=match):
        probe.parse_params(kind, env)


# ---------------------------------------------------------------- patching

def patched(tmp_path, text: str, name: str, value: int, others: dict[str, str] | None = None):
    (tmp_path / "train_gpt.py").write_text(text)
    for fname, t in (others or {}).items():
        (tmp_path / fname).write_text(t)
    f, old = probe.set_hyperparameter(tmp_path, name, value)
    return f, old, f.read_text()


@pytest.mark.parametrize("name,value,old,line", [
    ("val_loss_every", 100, "250",
     "    val_loss_every: int = 100  # every how many steps to evaluate val loss? 0 for only at the end"),
    ("val_loss_every", 0, "250",
     "    val_loss_every: int = 0  # every how many steps to evaluate val loss? 0 for only at the end"),
    ("num_iterations", 1800, "2210", "    num_iterations: int = 1800  # number of iterations to run"),
])
def test_patches_the_real_train_gpt(tmp_path, name, value, old, line):
    f, got_old, new = patched(tmp_path, REAL, name, value)
    assert f.name == "train_gpt.py" and got_old == old
    before, after = REAL.splitlines(), new.splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 1  # one line, and only that one
    assert changed[0][1] == line
    ast.parse(new)
    # the uses (args.num_iterations, args.val_loss_every > 0) are untouched
    assert new.count(f"args.{name}") == REAL.count(f"args.{name}") > 0


def test_patch_keeps_an_env_style_assignment_working(tmp_path):
    text = 'class H:\n    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))  # eval\n'
    _, old, new = patched(tmp_path, text, "val_loss_every", 100)
    assert old == 'int(os.environ.get("VAL_LOSS_EVERY", 1000))'
    assert new == "class H:\n    val_loss_every = 100  # eval\n"


def test_patch_finds_hyperparameters_in_a_sibling_module(tmp_path):
    _, _, _ = patched(tmp_path, "from hp import Hyperparameters\nprint(args.val_loss_every)\n", "val_loss_every",
                      100, {"hp.py": "class Hyperparameters:\n    val_loss_every: int = 125\n"})
    assert (tmp_path / "hp.py").read_text() == "class Hyperparameters:\n    val_loss_every: int = 100\n"


def test_patch_not_found_is_an_infra_error(tmp_path):
    (tmp_path / "train_gpt.py").write_text("x = args.val_loss_every\nif step == val_loss_every:\n    pass\n")
    with pytest.raises(probe.InfraError, match="found no assignment of `val_loss_every = …` in train_gpt.py"):
        probe.set_hyperparameter(tmp_path, "val_loss_every", 100)


def test_two_assignments_are_an_infra_error(tmp_path):
    (tmp_path / "train_gpt.py").write_text("class H:\n    num_iterations: int = 2000\nnum_iterations = 5\n")
    with pytest.raises(probe.InfraError, match=r"found 2 assignments \(train_gpt.py line 2, train_gpt.py line 3\)"):
        probe.set_hyperparameter(tmp_path, "num_iterations", 1800)
    assert "2000" in (tmp_path / "train_gpt.py").read_text()  # left as it was


def test_read_int():
    assert probe.read_int(HERE / "fixtures" / "train_gpt_mtpmed.py", "num_iterations") == 2210
    assert probe.read_int(HERE / "fixtures" / "missing.py", "num_iterations") is None


# ---------------------------------------------------------------- reading a run's output

def log(ms: float, total: int, *, upto: int, val_every: int, loss=lambda k: 4 + 3 * math.exp(-k / 300),
        nan_at: int = -1) -> str:
    """Console output in train_gpt.py's format: validation at step 0 and every val_every steps (before that step's
    training), training lines after each step."""
    out = ["import torch  # the script prints its own code first", "=" * 100]
    t = 0.0
    for step in range(0, upto + 1):
        if step == total or (val_every > 0 and step % val_every == 0):
            v = "nan" if step == nan_at else f"{loss(step):.4f}"
            out.append(f"step:{step}/{total} val_loss:{v} train_time:{t:.0f}ms step_avg:{t / max(step, 1):.2f}ms")
        if step == total:
            break
        t += 3000 if step < 10 else ms
        out.append(f"step:{step + 1}/{total} train_time:{t:.0f}ms step_avg:{t / (step + 1):.2f}ms")
    return "\n".join(out) + "\n"


def test_measure_loss_run_reads_every_100th_validation_and_the_slope():
    m = probe.measure_loss_run(log(480, 2250, upto=320, val_every=100), 300)
    assert m["val_steps"] == [100, 200, 300]  # step 0's validation isn't one of them
    assert m["val_loss"] == [round(4 + 3 * math.exp(-k / 300), 4) for k in (100, 200, 300)]
    assert m["ms_per_step"] == pytest.approx(480.0)
    assert m["steps"] == 2250


def test_measure_loss_run_with_steps_not_a_multiple_of_100():
    m = probe.measure_loss_run(log(480, 2250, upto=260, val_every=100), 250)
    assert m["val_steps"] == [100, 200]
    assert m["ms_per_step"] == pytest.approx(480.0)


def test_a_missing_validation_line_is_an_error():
    with pytest.raises(ValueError, match="no val_loss line at step"):
        probe.measure_loss_run(log(480, 2250, upto=320, val_every=250), 300)


def test_loss_run_done_waits_for_the_last_validation():
    done = probe.loss_run_done(300)
    full = log(480, 2250, upto=305, val_every=100)
    cut = full[:full.index("step:300/2250 val_loss")]  # training line 300 printed, its validation not yet
    assert not done(cut) and done(full)
    assert done(log(480, 2250, upto=305, val_every=0))  # two steps past with no validation: stop, and report it


def test_final_line_and_nan():
    text = log(480, 1790, upto=1790, val_every=0)
    assert probe.final_line(text) == (1790, round(4 + 3 * math.exp(-1790 / 300), 4), pytest.approx(3000 * 10 + 480 * 1780))
    r = probe.full_result(text, 1750)
    assert set(r) == {"steps", "train_time_s", "num_iterations", "val_loss", "margin_to_target", "ms_per_step", "caveats"}
    assert r["steps"] == 1790 and r["ms_per_step"] == pytest.approx(480.0)
    bad = probe.full_result(log(480, 1790, upto=1790, val_every=0, nan_at=1790), None)
    assert "val_loss" not in bad and "diverged" in bad["critique"][0]
    json.dumps(bad, allow_nan=False)


def test_loss_result_shape_and_delta():
    p = probe.measure_loss_run(log(480, 2250, upto=310, val_every=100), 300)
    c = probe.measure_loss_run(log(500, 2200, upto=310, val_every=100, loss=lambda k: 3.99 + 3 * math.exp(-k / 300)), 300)
    r = probe.loss_result(p, c, 300)
    assert set(r) == {"val_steps", "parent_val_loss", "candidate_val_loss", "val_loss_delta", "ms_per_step",
                      "parent_ms_per_step", "step_delta_pct", "caveats"}
    assert r["val_steps"] == [100.0, 200.0, 300.0]
    assert r["val_loss_delta"] == pytest.approx(-0.01)
    assert r["step_delta_pct"] == pytest.approx(500 / 480 * 100 - 100, abs=0.01)
    assert "one seed" in r["caveats"] and "doesn't fix the final value" in r["caveats"]


# ---------------------------------------------------------------- main() with a fake torchrun

# Stands in for torchrun: like the real one, it starts its worker in a session of its own and waits for it. The worker
# reads val_loss_every and num_iterations the way train_gpt.py would (from the patched copy), runs the same loop shape
# (validation before a step's training, the last step only validates), and with HANG = 1 keeps running after its
# last line, as a real run keeps training past the probe's step.
FAKE_TORCHRUN = textwrap.dedent("""\
    #!{python}
    import math, os, re, subprocess, sys, time
    if os.environ.get("FAKE_WORKER") != "1":
        w = subprocess.Popen([sys.executable, __file__], env={{**os.environ, "FAKE_WORKER": "1"}},
                             start_new_session=True)
        open("worker.pid", "w").write(str(w.pid))
        sys.exit(w.wait())
    assert "HARNESS_RESULT_FILE" not in os.environ and os.environ["TORCHINDUCTOR_CACHE_DIR"]
    src = open("train_gpt.py").read()
    knob = lambda k: float(re.search(r"^ *" + k + r"(?: *: *int)? = (-?[\\d.]+)", src, re.M).group(1))
    ms, crash, shift, hang, slow = knob("MS_PER_STEP"), int(knob("CRASH_AT")), knob("LOSS_SHIFT"), knob("HANG"), knob("SLOW")
    every, total = int(knob("val_loss_every")), int(knob("num_iterations")) + int(knob("EXTENSION"))
    t = 0.0
    for step in range(total + 1):
        if step == total or (every > 0 and step % every == 0):
            loss = 4 + 3 * math.exp(-step / 300) + shift
            print(f"step:{{step}}/{{total}} val_loss:{{loss:.4f}} train_time:{{t:.0f}}ms step_avg:{{t / max(step, 1):.2f}}ms", flush=True)
        if step == total:
            break
        if step + 1 == crash:
            print("[rank0]: Traceback (most recent call last):\\n[rank0]:   File \\"train_gpt.py\\", line 9, in <module>\\n"
                  "[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered", file=sys.stderr, flush=True)
            sys.exit(1)
        if slow and step > 60:
            time.sleep(slow)
        t += 3000 if step < 10 else ms
        print(f"step:{{step + 1}}/{{total}} train_time:{{t:.0f}}ms step_avg:{{t / (step + 1):.2f}}ms", flush=True)
    while hang:
        time.sleep(1)
    """)

CODE = textwrap.dedent("""\
    MS_PER_STEP = {ms}
    CRASH_AT = {crash}
    LOSS_SHIFT = {shift}
    HANG = {hang}
    SLOW = {slow}
    EXTENSION = 40
    class Hyperparameters:
        num_iterations: int = {iters}  # number of iterations to run
        val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    """)


def code(**kw) -> str:
    return CODE.format(**{"ms": 440, "crash": 0, "shift": 0.0, "hang": 1, "slow": 0, "iters": 1750, **kw})


def diff(old: str, new: str) -> str:
    o, n = old.splitlines(), new.splitlines()
    return ("diff --git a/subjects/modded-nanogpt/train_gpt.py b/subjects/modded-nanogpt/train_gpt.py\n"
            "--- a/subjects/modded-nanogpt/train_gpt.py\n+++ b/subjects/modded-nanogpt/train_gpt.py\n"
            f"@@ -1,{len(o)} +1,{len(n)} @@\n" + "".join(f"-{x}\n" for x in o) + "".join(f"+{x}\n" for x in n))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    tr = bin_ / "torchrun"
    tr.write_text(FAKE_TORCHRUN.format(python=sys.executable))
    tr.chmod(tr.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    fetched: list[str] = []
    monkeypatch.setattr(bench, "gpu_info", lambda: "NVIDIA H100 80GB HBM3, 580.159.04")
    monkeypatch.setattr(bench, "fetch", lambda name, dest: fetched.append(name) or (dest / name).write_bytes(b"") or dest / name)
    monkeypatch.setattr(bench, "cuda_home", lambda root: str(root))
    monkeypatch.setattr(bench, "prefetch_kernels", lambda code: None)
    monkeypatch.setattr(probe, "SAVE_DIR", str(tmp_path / "saved"))
    monkeypatch.delenv("HARNESS_EVAL_TIMEOUT_S", raising=False)
    out = tmp_path / "result.json"
    monkeypatch.setattr(probe, "RESULT_FILE", str(out))

    def go(kind: str, parent: str | None, cand: str, **params) -> dict | None:
        subj = tmp_path / "subjects" / "modded-nanogpt"
        subj.mkdir(parents=True, exist_ok=True)
        (subj / "train_gpt.py").write_text(cand)
        changes = tmp_path / "changes.diff"
        changes.write_text(diff(parent, cand) if parent else "")
        monkeypatch.setenv("HARNESS_SUBJECTS_DIR", str(tmp_path / "subjects"))
        monkeypatch.setenv("HARNESS_CHANGES_FILE", str(changes))
        monkeypatch.setenv("PROBE_KIND", kind)
        monkeypatch.setenv("HARNESS_PROBE", "1")
        for k, v in params.items():
            monkeypatch.setenv(f"PROBE_{k.upper()}", str(v))
        probe.main()
        assert (subj / "train_gpt.py").read_text() == cand  # only the temporary copy is patched
        return json.loads(out.read_text()) if out.exists() else None

    go.fetched = fetched
    go.saved = tmp_path / "saved"
    return go


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_loss_vs_parent_end_to_end(setup, capsys):
    r = setup("loss-vs-parent", code(ms=440), code(ms=462, shift=-0.02), steps=300)
    assert set(r) == {"val_steps", "parent_val_loss", "candidate_val_loss", "val_loss_delta", "ms_per_step",
                      "parent_ms_per_step", "step_delta_pct", "caveats", "wall_s"}
    assert r["val_steps"] == [100.0, 200.0, 300.0]
    assert r["parent_val_loss"] == [round(4 + 3 * math.exp(-k / 300), 4) for k in (100, 200, 300)]
    assert r["val_loss_delta"] == pytest.approx(-0.02)
    assert (r["ms_per_step"], r["parent_ms_per_step"]) == (462.0, 440.0)
    stages = [line[7:] for line in capsys.readouterr().out.splitlines() if line.startswith("stage: ")]
    assert stages == ["rebuilding the parent", "downloading data", "compiling parent", "training parent",
                      "compiling candidate", "training candidate"]
    assert len(setup.fetched) == 1 + probe.TRAIN_SHARDS["loss-vs-parent"]
    # the run was stopped after step 300, and torchrun's worker (in a session of its own) with it
    for label in ("parent", "candidate"):
        assert "step:300/1790 val_loss" in (setup.saved / f"{label}.out").read_text()


def test_loss_vs_parent_stops_torchruns_worker(setup, tmp_path):
    c = tmp_path / "code"
    c.mkdir()
    (c / "train_gpt.py").write_text(code())
    r = probe.run(c, tmp_path, tmp_path / "cache", "t", 60, probe.loss_run_done(200))
    assert r["reached"] >= 200 and not r["finished"]
    worker = int((c / "worker.pid").read_text())
    for _ in range(50):
        if not alive(worker):
            break
        time.sleep(0.1)
    assert not alive(worker)


def test_loss_vs_parent_candidate_crash_is_a_critique(setup):
    r = setup("loss-vs-parent", code(), code(crash=150), steps=300)
    assert set(r) == {"val_steps", "parent_val_loss", "parent_ms_per_step", "caveats", "critique", "wall_s"}
    assert r["critique"][0].startswith("[FAIL] on one H100, ./train_gpt.py exited with code 1 at training step 149, "
                                       "before training step 300 and its validation. The parent's code got there")
    assert "illegal memory access" in r["critique"][0] and "First traceback" in r["critique"][0]


def test_loss_vs_parent_parents_crash_is_not_the_candidates(setup):
    with pytest.raises(probe.InfraError, match="the parent's code didn't get to training step 300"):
        setup("loss-vs-parent", code(crash=50), code(), steps=300)
    assert not Path(probe.RESULT_FILE).exists()


def test_a_pattern_not_found_ends_before_any_download(setup):
    cand = code().replace("val_loss_every", "val_every")
    with pytest.raises(probe.InfraError, match="can't set val_loss_every = 100"):
        setup("loss-vs-parent", code().replace("val_loss_every", "val_every").replace("440", "430"), cand)
    assert setup.fetched == [] and not Path(probe.RESULT_FILE).exists()


def test_loss_vs_parent_on_the_base_node(setup):
    r = setup("loss-vs-parent", None, code())
    assert set(r) == {"critique", "wall_s"} and "base node" in r["critique"][0]
    assert setup.fetched == []


def test_no_gpu_gives_no_result(setup, monkeypatch):
    monkeypatch.setattr(bench, "gpu_info", lambda: "")
    with pytest.raises(probe.InfraError, match="needs an H100"):
        setup("full-run-1gpu", None, code())


def test_full_run_end_to_end_with_num_iterations(setup, capsys):
    r = setup("full-run-1gpu", None, code(ms=470, hang=0), num_iterations=1800)
    assert set(r) == {"val_loss", "margin_to_target", "steps", "num_iterations", "train_time_s", "ms_per_step", "caveats", "wall_s"}
    assert (r["steps"], r["num_iterations"]) == (1840, 1800)
    assert r["val_loss"] == round(4 + 3 * math.exp(-1840 / 300), 4)
    assert r["ms_per_step"] == pytest.approx(470.0)
    assert r["train_time_s"] == pytest.approx((3000 * 10 + 470 * 1830) / 1000)
    out = (setup.saved / "candidate.out").read_text()
    assert out.count("val_loss") == 1  # validation only at the end (val_loss_every = 0 in the copy)
    stages = [line[7:] for line in capsys.readouterr().out.splitlines() if line.startswith("stage: ")]
    assert stages == ["downloading data", "compiling candidate", "training candidate", "validating candidate"]


def test_full_run_without_num_iterations_uses_the_codes(setup):
    r = setup("full-run-1gpu", code(), code(hang=0, iters=1700))
    assert (r["steps"], r["num_iterations"]) == (1740, 1700)


def test_full_run_crash_is_a_critique(setup):
    r = setup("full-run-1gpu", None, code(hang=0, crash=900))
    assert set(r) == {"reached_step", "steps", "num_iterations", "ms_per_step", "caveats", "critique", "wall_s"}
    assert (r["reached_step"], r["steps"], r["num_iterations"]) == (899, 1790, 1750)
    assert "exited with code 1 at training step 899" in r["critique"][0]
    assert "illegal memory access" in r["critique"][0]


def test_full_run_that_doesnt_finish_in_time_is_a_critique(setup, monkeypatch):
    monkeypatch.setattr(probe, "REPORT_RESERVE_S", 0)
    monkeypatch.setenv("HARNESS_EVAL_TIMEOUT_S", "4")  # the eval's limit, below max_wall_s
    r = setup("full-run-1gpu", None, code(hang=0, slow=0.05))
    assert "didn't get to the end of its run and the final validation within" in r["critique"][0]
    assert 60 < r["reached_step"] < 1790


def test_an_infra_error_exits_zero_with_no_result(tmp_path):
    """Run as a script: exit 0 and no result file, which Kupanga reads as a broken eval, not a crash."""
    out = tmp_path / "result.json"
    env = {**os.environ, "HARNESS_RESULT_FILE": str(out), "HARNESS_SUBJECTS_DIR": str(tmp_path / "subjects"),
           "HARNESS_CHANGES_FILE": str(tmp_path / "changes.diff"), "PATH": "/usr/bin:/bin",
           "PROBE_KIND": "full-run-1gpu", "HARNESS_PROBE": "1"}
    (tmp_path / "subjects" / "modded-nanogpt").mkdir(parents=True)
    (tmp_path / "subjects" / "modded-nanogpt" / "train_gpt.py").write_text(code())
    p = subprocess.run([sys.executable, str(EVAL / "probe.py")], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "not the candidate's failure, no result: needs an H100" in p.stdout
    assert not out.exists()
    assert not any(line.lstrip().startswith("{") for line in p.stdout.splitlines())


def test_a_bad_parameter_exits_zero_with_no_result(tmp_path):
    out = tmp_path / "result.json"
    env = {**os.environ, "HARNESS_RESULT_FILE": str(out), "PROBE_KIND": "loss-vs-parent", "PROBE_STEPS": "9000"}
    p = subprocess.run([sys.executable, str(EVAL / "probe.py")], env=env, capture_output=True, text=True)
    assert p.returncode == 0 and "PROBE_STEPS=9000 is outside 100–500" in p.stdout and not out.exists()
