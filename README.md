# speedrun-probe

A Kupanga probe eval for the modded-nanogpt speedrun: bounded experiments the agent runs on one H100 during its
turn, and gets numbers back from. The self-check (speedrun-perf) tells the agent "doesn't crash, isn't slower";
it never says whether an idea works. In `nanogpt-jumps-selfcheck`, mtp-minimal cut 55 steps before measuring
anything and finished +2.17 s against its parent with 0.0074 of loss headroom unused. A probe answers the two
questions it lacked: is the candidate's loss ahead of the parent's at the same step, and what val loss does a
whole run reach at a chosen step count. Design: `notes/gpu-probes-design.md`.

A procs eval (`type = "procs"`, formerly `type = "probe"`) is never a gate or a score and never runs after a turn. Its numbers are on one
GPU, so they aren't the benchmark's.

| File | What it is |
|---|---|
| `eval.toml` | `type = "procs"`, `python probe.py`, the two kinds as experiment procs (`[eval.procs.<kind>]`) with their parameters, `max_wall_s`, stages and outputs |
| `job.toml` | `python:3.12`, 1×H100 (spot, 3 retries), 16+ CPUs, 64+ GB, `timeout_s` 2100 (the largest `max_wall_s`) |
| `requirements.txt` | speedrun-8xh100's pinned stack (a copy, as in speedrun-perf) |
| `bench.py` | speedrun-8xh100's `bench.py` (a copy): data, CUDA headers, hub kernels, parent rebuild, `failure_excerpt` |
| `probe.py` | Reads the kind and parameters, patches a temporary copy of the code, runs it, writes the result |
| `tests/` | `python -m pytest tests`, no GPU (a fake `torchrun` stands in); `fixtures/train_gpt_mtpmed.py` is a real `train_gpt.py` of the subject, for the patch |

## Interface

Kupanga starts `python probe.py` with:
- `PROBE_KIND`: `loss-vs-parent` or `full-run-1gpu`;
- `PROBE_<PARAM>`, the parameter's name upper-cased (`PROBE_STEPS`, `PROBE_NUM_ITERATIONS`); an absent one takes
  its default, or none if it is optional;
- `HARNESS_PROBE=1`, and the usual `HARNESS_*` variables (`HARNESS_SUBJECTS_DIR`, `HARNESS_CHANGES_FILE`,
  `HARNESS_RESULT_FILE`, `HARNESS_EVAL_TIMEOUT_S`).

`probe.py` reads the parameters' bounds and defaults and each kind's `max_wall_s` from `eval.toml`, so they are
written once. It stops itself at the kind's `max_wall_s` (or `HARNESS_EVAL_TIMEOUT_S`, if smaller), keeping 60 s
to stop the run and report.

The result is one JSON object, written to `$HARNESS_RESULT_FILE` and printed, as speedrun-perf does. Its fields
are the kind's declared outputs, plus `critique` (a list of strings) when there are no candidate numbers, and
`wall_s`. While it runs it prints `stage: <name>` lines from the kind's `stages`. It also echoes each code's step lines (every 10th
training step and each validation) as `<parent|candidate> step:N/M [val_loss:X ]train_time:Tms`, which both kinds'
`progress_regex` reads for the agent's `inspect(id, "progress")`.

## The kinds

### loss-vs-parent (`steps`: 100–500, default 300)

1. Rebuilds the parent's code with `bench.parent_code` (reverse-applies `$HARNESS_CHANGES_FILE`). The base node
   has no parent: the result is a `critique` saying so, with nothing measured.
2. In a temporary copy of each code, sets `val_loss_every` to 100 (see Patching).
3. Downloads the val shard and 3 training shards, and fetches the hub kernels.
4. Runs the **parent first**, then the candidate, each with `torchrun --standalone --nproc_per_node=1 train_gpt.py`
   until training step `steps` and its validation have printed, then stops torchrun and its worker (which runs in
   a session of its own; speedrun-perf's `stop()`). `num_iterations` isn't touched, so each code runs its own
   LR schedule, unmodified, and is cut off at the step. Before each run the GPU must be free. Both runs share one
   Inductor and Triton cache, so only what the candidate changed compiles twice.
5. Reads each run's `step:k/N val_loss:X` lines at k = 100, 200, … and fits ms/step over the
   `step:k/N train_time:Xms` lines from step 50 to `steps`, as speedrun-perf does.

| Output | |
|---|---|
| `val_steps` | 100, 200, … up to the last multiple of 100 not past `steps` |
| `parent_val_loss`, `candidate_val_loss` | val loss at those steps (all 10.5M validation tokens; a non-finite value is null, with a critique) |
| `val_loss_delta` | candidate − parent at the last of `val_steps`; negative means the candidate is ahead |
| `ms_per_step`, `parent_ms_per_step` | train_time per step, the slope over steps 50 to `steps`, ms, one H100 |
| `step_delta_pct` | candidate ms/step ÷ parent − 1, percent |
| `caveats` | the noise warning below, always |

**Same seed:** the subject's `train_gpt.py` seeds nothing itself, so both codes start from torch's default seed
(the same initial weights, if the candidate builds the same parameters in the same order) and read the data in the
same order. GPU kernels aren't bit-deterministic, so two runs of the same code still differ slightly.

**Validation tokens aren't reduced.** A validation pass is forward only over 10.5M tokens, roughly 10 s on one
H100 (estimated, not yet measured), so five of them cost under a minute per code. Keeping the benchmark's full
validation set keeps that source of noise out.

### full-run-1gpu (`num_iterations`: 1500–2600, optional)

`baseline = "parent"`: this kind runs only the code it is given, so Kupanga also runs it on the parent's code (the
node's starting code) with the same `num_iterations`, the first time a node asks for those params, and the agent's
report gives candidate, parent and delta for `val_loss`, `steps` and the timings. That first request holds twice the
time limit below; later ones reuse the parent's numbers. loss-vs-parent is `baseline = "in-job"`: it measures the
parent itself every time (it doesn't read `$HARNESS_BASELINE_FILE` or report a `parent` object).

`repeats = 4`: the agent may ask for `repeats` 2 to 4, and Kupanga then runs that many copies of this kind in
parallel on separate H100s (and as many of the parent's) and reports the mean, the noise divided by √n and each
run's val loss. It costs n times the GPU-minutes in the same wall time; probe.py never sees the parameter. Left
out, `num_iterations` is the code's own value, and a parent measured that way (it reports `num_iterations`) also
serves a later request that names the same value.

`margin_to_target` (3.28 − val_loss) and `num_iterations_for_target` turn the loss into the decision the
benchmark makes. The benchmark counts a time when its 8 runs average below 3.28 at p < 0.01, which passes 80% of
the time at a mean of about 3.2779. `num_iterations_for_target` is the step count that would put this run's loss
there, at 5.9e-5 val_loss a step. Both are linear in the loss, so Kupanga's mean over repeats is the answer for
the mean loss, and its noise (25 steps for one run) shrinks by √n. One run can't place a step count; 4 repeats
place it to about ±13 steps. speedrun-target checks the committed candidate the same way with 8 runs after the
turn.

1. In a temporary copy of the candidate's code, sets `val_loss_every` to 0 (validation only after the last step,
   which saves about 10 passes) and, if given, `num_iterations` to the parameter.
2. Downloads the val shard and 10 training shards (2640 steps of 262k tokens is 692M), fetches the hub kernels.
3. Runs the whole training run with `torchrun --nproc_per_node=1`: one GPU, so `train_gpt.py` takes 8
   gradient-accumulation micro-steps per step, the same math as 8×H100. It stops when the final validation line
   has printed.

| Output | |
|---|---|
| `val_loss` | val loss after the last step |
| `steps` | the planned total steps (`num_iterations` plus the code's own extension, 40 in the jumps records) |
| `num_iterations` | the parameter, or the code's own value when it is a literal |
| `train_time_s` | train_time of the final line, seconds: one GPU, about 8× the benchmark's, a relative signal only |
| `ms_per_step` | the slope over steps 50 to the end, ms |
| `reached_step` | only when the run didn't finish |
| `caveats` | the noise warning, always |

A record's own checks on `num_iterations` (mtp-medium asserts that its embedding split falls on an even step) still
apply: if the parameter breaks one, that is a crash of the candidate's code, reported as one.

## Patching

The probe never edits the code under test; it patches its own temporary copy. `set_hyperparameter` finds the
single assignment `name = …` or `name: type = …` at the start of a line (the field in the record's
`Hyperparameters` dataclass: `val_loss_every: int = 250`, `num_iterations: int = 2240` in the jumps records), in
`train_gpt.py` and then in the record's other top-level modules, and replaces its right-hand side, keeping the
comment. Uses like `args.val_loss_every` aren't assignments and aren't touched. If there isn't exactly one
assignment, the probe ends with **no result** and says what it found where; it never runs unpatched code. The patch
happens before any download, so a missing pattern costs no GPU time beyond the machine's start. If the run still
doesn't print the expected validation lines, that is also no result.

## Failures

- **The candidate's code** crashes, or doesn't get to its goal in the time it has: the result carries a `critique`
  with why, the step reached (and its ms/step, if measurable) and its first traceback (`bench.failure_excerpt`).
  loss-vs-parent keeps the parent's numbers in that result. In loss-vs-parent a candidate timeout counts as the
  candidate's only when it had at least 1.25× the parent's wall time plus 60 s; otherwise the probe ran out of time
  (no result).
- **Anything else** (a bad parameter, no H100, a download, a kernel fetch, rebuilding the parent, the parent's own
  run, a patch pattern not found, out of time): the probe prints why and exits 0 **without a result**, which
  Kupanga reads as a broken eval rather than the candidate's crash.

## Cost and time

From speedrun-perf's run on one H100 (managed job 2259): a code compiles in about 400 s to its first step
(validation at step 0 included); with the shared cache, a candidate that changes the model reached step 250 in
170–240 s, about 50–120 s of compiling; a step costs 0.45–0.50 s. Setup (data, git, kernels) is about 90 s.

**loss-vs-parent, `max_wall_s` 1500.** At 500 steps: setup 90 + parent (400 compile + 500 × 0.48 = 240 +
5 validations × ~10 = 50, so 690) + candidate (~120 compile + 250 + 50, so 420) + GPU-free waits and stops ~30 ≈
1230 s. A candidate whose graphs all recompile adds about 280 s (≈ 1510 s), and a slower step adds 10–25 s per code.
1200 s would cut a typical 500-step probe short, so the limit is 1500 s (25 GPU-minutes held, about 20 charged).
At the default 300 steps it is about 1000 s. Roughly $1 at $3 per H100-hour.

**full-run-1gpu, `max_wall_s` 2100.** At `num_iterations` = 2600 (2640 steps): setup 90 + compile 400 + 2640 ×
0.48 = 1270 (1320 at 0.50 s) + the final validation and its compile ~70 ≈ 1880 s; at the jumps records' own
~2250 steps, about 1700 s. That leaves about 10% to spare at the top of the range; a candidate much slower than
0.5 s a step at 2600 iterations may not finish (a critique, with how far it got). About $1.60.

Both holds count against Kupanga's probe budgets (30 GPU-minutes a turn by default in the design): a 1500 s hold
leaves no room for a second loss-vs-parent until the first settles, and a 2100 s (35-minute) full-run-1gpu hold
is more than the default per-turn cap.

Each kind also has a `[wall]` formula in eval.toml, so a smaller request holds less and its job is killed at its own limit:
- loss-vs-parent: 1000 s + 1.2 s × steps, at most 1500 (300 steps: 1360 s; job 2384 took 784 s);
- full-run-1gpu: 500 s + 0.5 s × num_iterations, at most 1800 (measured 1430–1490 s at 2380 steps on 2026-11-03).
Without a value for a parameter the formula uses, a probe holds `max_wall_s`.

## Checked on one H100

Managed job 2384 (GCP a3-highgpu-1g spot), loss-vs-parent at 300 steps on nanogpt-jumps-stacked's mtp-minimal
against its parent, outside Kupanga with the same environment variables: exit 0, every stage printed, the result
had exactly the declared outputs, 784 s of wall time (parent 465 s including its compile, candidate 291 s with the
shared cache) against the 1500 s hold.

| | Parent | Candidate |
|---|---|---|
| val loss at 100 / 200 / 300 | 4.6788 / 4.1885 / 4.0209 | 5.0270 / 4.5602 / 4.3863 |
| ms/step | 451.4 | 496.7 (+10.0%) |

The step cost matches speedrun-perf on the same case (+10.4%) and 8×H100 (+10.5%). The loss gap does not
predict the end (see Noise).

## Noise

- **One seed.** Each code trains once. Differences of a few thousandths in val loss can be noise (±0.0015 a run,
  ±0.0021 on a delta: backout-minimal-3's two runs of the same parent were 0.0014 apart). full-run-1gpu's `repeats`
  averages up to 3 runs of each code.
- **Early loss doesn't fix the final value.** At steps 100–500 most of the LR schedule, its cooldown included, is
  still ahead. A lead over the parent at 300–500 is a signal, not a result: a method can lead early and finish
  level, or the other way round. A method whose effect changes over training misleads most: on the H100 check
  below, multi-token prediction (its extra loss terms take weight from the next-token loss early and fade later)
  trailed its parent by 0.37 at step 300, yet that code finished the full 8×H100 run within about 0.01 of it. For
  such a method, use full-run-1gpu.
- **One GPU isn't 8×H100.** ms/step and train_time have no multi-GPU communication or optimizer sharding.
  speedrun-perf found one H100 overstated mtp-detailed's step cost (+2.7% vs +0.8% on 8×H100). Compare 1-GPU
  numbers with each other only.

## Settings

| Variable | Default | |
|---|---|---|
| `SPEEDRUN_PROBE_TRAIN_SHARDS` | 3 / 10 | Training shards for loss-vs-parent / full-run-1gpu |
| `SPEEDRUN_PROBE_SAVE_DIR` | unset | Keeps each run's output (`parent.out`, `candidate.out`) there, for `--parse`; for debugging |

Local check, no GPU: `python probe.py --parse loss-vs-parent <parent.out> <candidate.out> [steps]` or
`python probe.py --parse full-run-1gpu <candidate.out>`.
