# AgentFact

Official repository for the paper:

**RW-Post: Auditable Evidence-Grounded Multimodal Fact-Checking in the Wild**

Published at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2026 — please refer to the paper for detailed methodology and experimental settings (see [Citation](#citation) below).

A multi-agent, LLM-based framework for multimodal misinformation fact-checking. Given a social media post (text + image) and its claim, AgentFact plans a verification strategy, gathers text/image evidence, judges source credibility, and produces an explainable veracity verdict with evidence-ID citations.

## 📢 Updates

### RW_Post Dataset Released

The RW_Post dataset has been uploaded. The non-image portion (claims, post text, evidence, labels) is included in this repo under `RW_Post_dataset/`. A small self-contained demo sample with images also ships under `Dataset_RW-Post/demo/` so you can smoke-test the pipeline right away.

The complete dataset, including post images, is available on:
- Google Drive 📎 https://drive.google.com/file/d/1bo4zyrJX_3iLziMqVL5z_urqQ2pddr3k/view?usp=sharing
- IEEE DataPort 📎 https://dx.doi.org/10.21227/vasq-rf11

#### Terms of Use

By accessing or using this dataset, you agree to comply with the terms described in [DATA_LICENSE.md](DATA_LICENSE.md).

Please refer to [dataset_instruction.md](dataset_instruction.md) for the data schema and recommended evaluation settings (closed-book / evidence-bounded / open-web).

### Code Released

The AgentFact pipeline code is now available — see below for the code structure and how to run it.

## Agents

The pipeline is organized around five cooperating agents:

| Agent | Role | Code |
|---|---|---|
| Strategy Planning | Designs a validation plan, a list of statements to verify, and search intents | `call_Strategy_generator` in [agent_model.py](agent_model.py) |
| Text Evidence Retrieval & Validation | Generates search queries, searches the web, filters/summarizes results, and rates source reliability (reliable/unreliable/satire/unsure) | `call_Query_Site_generator`, `call_evidence_source_with_link_judger_detailed` in [agent_model.py](agent_model.py); `query_search`/`text_search`/`evidence_extraction`/`get_evidence` in [query_evidence.py](query_evidence.py) and [retrieval_imagehash_with_google_vision.py](retrieval_imagehash_with_google_vision.py) |
| Image Retrieval & Analysis | Reverse image search, then classifies the image relationship, detects tampering, and checks caption consistency | `visual_search` in [retrieval_imagehash_with_google_vision.py](retrieval_imagehash_with_google_vision.py); `call_image_similarity_type_and_manipulation_judger`, `call_image_miscaption_detector` in [agent_model.py](agent_model.py) |
| Reasoning | Rephrases the claim, reasons step-by-step over accumulated evidence, and scores evidence sufficiency | `call_Middle_Reasoner`, `call_Middle_Reasoner_with_Source_Judgment` in [agent_model.py](agent_model.py) |
| Explanation Generation | Produces the final veracity label, reasoning summary, and evidence-ID-cited key points | `call_Explainer_3_class_aligned` in [agent_model.py](agent_model.py) |

`main_workflow.py` orchestrates these agents in an iterative retrieve-reason loop per claim (see [Search modes](#search-modes) below for how much retrieval actually happens).

## Code structure

```
main_workflow.py       Entry point / CLI. Loads claims, runs the per-claim
                        agent loop, writes results.
agent_model.py          Prompts and LLM-calling functions for all five agents
                        (see table above for the mapping).
agent_utils.py          LLM client wrapper (OpenAIWrapper), JSON parsing/
                        repair helpers, message construction utilities.
dataload.py             Loads and validates claims from the ground-truth
                        JSONL format (checks images open correctly, extracts
                        claim/post/evidence fields).
configs.py              Global config: prompts/output/cache paths, API keys
                        read from environment variables (see Setup below).
query_evidence.py       Google Custom Search backend + generic webpage
                        scraping/domain-blocklist utilities.
retrieval_imagehash_with_google_vision.py
                        Text evidence pipeline (search -> filter -> scrape ->
                        LLM-summarize) and the Google Vision reverse-image-
                        search pipeline.
search/
  serper.py             Serper.dev web-search backend (the default backend
                         query_evidence.py's query_search() dispatches to).
  common.py             Query/Source/WebSource data structures.
config/
  globals.py             Loads config/api_keys.yaml; prompts interactively
                          for missing keys on first run.
  api_keys.example.yaml  Template for config/api_keys.yaml (gitignored).
prompts/
  evidence_extraction.md LLM prompt used by evidence_extraction() to filter/
                         summarize scraped webpage text down to what's
                         relevant to a given query.
run_workflow_batches.py Fans main_workflow.py out across parallel batches
                        for large runs.
run_dev.sh              Shell-script variant of the same batching idea.

RW_Post_dataset/        Non-image portion of the RW_Post dataset (see
                        "Updates" above for the full dataset with images).
Dataset_RW-Post/demo/   A small self-contained sample (5 labeled posts +
                        images) for smoke-testing the pipeline.
dataset_instruction.md  Dataset schema and recommended evaluation settings.
```

## Setup

```bash
pip install -r requirements.txt
```

### API keys

The pipeline needs an OpenAI key and a web-search backend key. Two independent config paths exist in this codebase — you'll typically only need the first:

1. **`.env`** (used by `configs.py` / `agent_utils.py`, the main pipeline entry points):
   ```bash
   cp .env.example .env
   # then fill in OPENAI_API_KEY (and optionally DEEPSEEK_API_KEY / GOOGLE_CSE_*)
   ```

2. **`config/api_keys.yaml`** (used by the default `serper` search backend in [search/serper.py](search/serper.py)):
   ```bash
   cp config/api_keys.example.yaml config/api_keys.yaml
   # then fill in serper_api_key (and openai_api_key if you rely on this path instead)
   ```
   Get a Serper API key at **[serper.dev](https://serper.dev/)** (free tier available). If you leave this file blank, [config/globals.py](config/globals.py) will prompt you for each key interactively on first run and save it back to this file.

Both `.env` and `config/api_keys.yaml` are gitignored — never commit real keys.

### Reverse image search (optional)

If you use the Google Vision-based image search, place your GCP service account credentials at `config/google_service_account_key.json` (also gitignored), and make sure both **billing** and the **[Cloud Vision API](https://console.cloud.google.com/apis/library/vision.googleapis.com)** are enabled on that GCP project. See Google's [Vision API setup guide](https://cloud.google.com/vision/docs/setup) for creating a project, enabling billing, and generating a service account key.

## Search modes

`main_workflow.py` supports three modes, set via the `search_mode` variable near the top of the file (not currently exposed as a CLI flag):

| Mode | Behavior |
|---|---|
| `"open_book"` (default) | Uses the claim's ground-truth `retrieved_evidence` field directly, with no live retrieval. Fastest/cheapest; useful for evaluating the reasoning/explanation agents in isolation. |
| `"close_book"` | No evidence at all — the model reasons purely from the claim and post content. |
| `"open_web"` | Full iterative retrieve-reason loop: generates search queries, retrieves web evidence via Serper, does reverse image search via Google Vision, and reasons over the results across multiple rounds (up to 10 steps) until confident. This is the mode described in the paper's Workflow section. |

To switch modes, edit:
```python
search_mode = "open_book"  # "open_web" or "close_book" or "open_book"
```

## Running the demo

A small self-contained sample (5 labeled posts + images) ships in `Dataset_RW-Post/demo/demo.jsonl` so you can smoke-test the pipeline without downloading the full dataset:

```bash
python main_workflow.py --input_file Dataset_RW-Post/demo/demo.jsonl --dataset rwpost
```

CLI arguments:

| Flag | Default | Meaning |
|---|---|---|
| `--input_file` | `Dataset_RW-Post/demo/demo.jsonl` | Path to a ground-truth JSONL file |
| `--dataset` | `rwpost` | Dataset name tag, used only for the output path |
| `--start_id` | `0` | Index of the first claim to process |
| `--end_id` | `-1` | Index to stop before (`-1` = process all remaining claims) |

For larger runs, `run_workflow_batches.py` (or `run_dev.sh`) fans this out across parallel batches — update `input_file`/`batch_size`/`max_workers` at the top of the script for your dataset size and hardware.

## Output

Results are written to `output/<dataset>/<mode_name>/<search_mode>/<start_id>/<model_version>/`:

- `output.jsonl` — one JSON record per successfully processed claim (`claim_id`, `post_info`, `context` with the full agent trace, and `final_answer` with the veracity label, reasoning, and confidence).
- `output_failed.jsonl` — claims where the pipeline could not produce a final answer after retries (e.g. persistent API failures), so they can be re-run later.
- `readable_json/<claim_id>.json` — a pretty-printed copy of each successful record.

Runs are resumable: `news_url`s already present in `output.jsonl` are skipped on the next run of the same command.

## Citation

If you use AgentFact or the RW-Post dataset in your research, please cite:

```bibtex
@InProceedings{Xu_2026_CVPR,
    author    = {Xu, Danni and Fan, Shaojing and Cheng, Harry and Kankanhalli, Mohan},
    title     = {RW-Post: Auditable Evidence-Grounded Multimodal Fact-Checking in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {10964-10973}
}
```

## License

The **code** in this repository is released under the MIT License — see [LICENSE](LICENSE).

The **RW_Post dataset** is released separately under the terms in [DATA_LICENSE.md](DATA_LICENSE.md).
