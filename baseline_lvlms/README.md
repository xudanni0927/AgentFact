# Controlled LVLM Benchmarking on RW-Post

A model-agnostic batch-evaluation framework for comparing LVLMs on RW-Post
under four controlled input settings:

| Setting  | Input                                  | Prompt template |
|----------|-----------------------------------------|------------------|
| `T`      | claim + post text                       | `prompts/fact_check_no_evi.md` |
| `T+I`    | claim + post text + post image          | `prompts/fact_check_no_evi.md` |
| `T+E`    | claim + post text + ground-truth evidence | `prompts/fact_check_evi.md` |
| `T+I+E`  | claim + post text + image + evidence    | `prompts/fact_check_evi.md` |

`T` and `T+I` are **closed-book**: the model answers from the claim/post/image
alone. `T+E` and `T+I+E` are **evidence-bounded**: the model is additionally
given the ground-truth fact-checking evidence snippets for that claim.

Output is written in the schema the project's `metrics_0802.py` /
`metrics_0802_jsonl.py` scripts expect, so those scripts can be pointed
directly at this framework's `output.jsonl` files.

## Adding a model

Implement `BaseVLM` (`models/base.py`) — just a `generate(prompt, image_path=None) -> str`
method — and register it in `MODEL_REGISTRY` in `run_baseline_eval.py`.
Model classes are imported lazily, so adding one doesn't require every other
backend's dependencies to be installed.

Backends included: `gpt-4o-mini`, `gemini-3-flash` (API-based), `qwen3-vl`,
`internvl3.5`, `qwen2-vl`, `llava-1.5` (local HF inference).

## Setup

This directory lives inside the AgentFact project and imports `dataload.py`
and `agent_utils.py` from the project root — make sure the root
`requirements.txt` is installed first (it provides `opencv-python-headless`,
`openai`, and other dependencies those files need).

**API-based models** (`gpt-4o-mini`, `gemini-3-flash`):
```bash
pip install requests google-genai python-dotenv
```
Set `OPENAI_API_KEY` and/or `GEMINI_API_KEY` in a `.env` file in this
directory (gitignored) or in your shell environment. Gemini must be on a
paid billing tier — the free tier's daily quota is too low for a full run.

**Local HF models** (`qwen3-vl`, `internvl3.5`, `qwen2-vl`, `llava-1.5`):
```bash
pip install -U transformers torch pillow opencv-python-headless
pip install qwen-vl-utils   # qwen2-vl only
```
Qwen3-VL requires a recent-enough `transformers` release for
`Qwen3VLForConditionalGeneration` support.

All backends use deterministic decoding (`temperature=0` for API models,
`do_sample=False` for local models) so results are reproducible and
comparable across models.

## Running

```bash
# from the AgentFact project root, or via the wrapper scripts below
python baseline_lvlms/run_baseline_eval.py --model gpt-4o-mini --setting T+E \
    --input_file RW_Post_dataset/demo/demo.jsonl
```

Key flags:
- `--model` / `--setting`: see `MODEL_REGISTRY` / the table above
- `--start_id` / `--end_id`: process a slice of the dataset (e.g. for a smoke test)
- `--evidence_k`: evidence-quantity ablation — randomly sample at most K
  evidence snippets per claim (fixed seed) instead of using everything;
  omit for the full-evidence baseline
- `--input_file`: defaults to `Dataset_RW-Post/test.jsonl`, the full local
  dataset used for the paper's results — **this path is not included in
  this repository**. For a quick runnable example use
  `RW_Post_dataset/demo/demo.jsonl` (ships with this repo, 5 claims); to
  reproduce the full paper numbers, obtain the complete RW-Post dataset
  separately and pass its path via `--input_file`.

Runs are resumable: each `(model, setting)` pair writes to its own
`results/<model>/<T|TI|TE|TIE>/output.jsonl`, and re-running skips claims
already present there.

Wrapper scripts (`run_*.sh`) loop over all three/four main settings for one
model each; `run_gpt4o_mini_evidence_ablation.sh` runs the evidence-quantity
sweep. They read a `PYTHON_BIN` env var (default: `python`) and, for the
local-GPU scripts, a `CUDA_VISIBLE_DEVICES` env var (default: `0`) — set
these to match your own environment and available GPU before running.

## Known model-output quirks

Some local models emit near-valid-but-not-quite JSON. For example, LLaVA-1.5
tends to escape underscores in field names (e.g. `label\_3class` instead of
`label_3class`), which is not a valid JSON escape sequence. `parse_model_json`
in `run_baseline_eval.py` normalizes this (`\_` -> `_`) before parsing, so it
does not need to be handled by callers. LLaVA-1.5 also occasionally emits a
degenerate, never-terminating `evidence_ids` list that runs until the
`max_new_tokens` cutoff, leaving the JSON truncated and unparseable; this is
a genuine model failure mode (observed at roughly a 1-2% rate), not a parsing
bug, and is recorded in `output_failed.jsonl` rather than silently retried
forever.

## Analysis

- `analyze_evidence_ratio.py`: pools the evidence-quantity ablation's K-tier
  outputs and re-buckets every record by its own evidence-coverage ratio
  (shown/total), to disentangle "how much evidence" from "what fraction of
  this claim's evidence."
- `plot_ablation_figures.py`: generates the modality-ablation and
  evidence-quantity figures used in the paper.
