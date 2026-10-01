"""
Unified batch runner for the T / T+I / T+E / T+I+E baseline LVLM comparison.

- T      : claim + post text only (prompts/fact_check_no_evi.md)
- T+I    : claim + post text + post image, no evidence (prompts/fact_check_no_evi.md)
- T+E    : claim + post text + ground-truth evidence snippets (prompts/fact_check_evi.md)
- T+I+E  : claim + post text + post image + evidence snippets (prompts/fact_check_evi.md)

Model backends are pluggable (see models/base.py) so new models can be added
without touching this file's batching/parsing/output logic. Output is written
in the schema metrics_0802.py / metrics_0802_jsonl.py already expect, so those
scripts can be pointed directly at this framework's output.

Usage:
    python run_baseline_eval.py --model gpt-4o-mini --setting T
    python run_baseline_eval.py --model gpt-4o-mini --setting T+E --start_id 0 --end_id 50
"""

import os
import sys
import json
import random
import argparse

# Make the AgentFact project root importable (for dataload.py) regardless of
# the caller's working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for models/

from dataload import extract_data_from_GT_jsonl  # noqa: E402

# Lazy import path -> class name, so selecting one model doesn't require
# every other model's dependencies (e.g. transformers, qwen_vl_utils,
# google-genai) to be installed. Register new backends here.
MODEL_REGISTRY = {
    "gpt-4o-mini": ("models.gpt4o_mini", "GPT4oMini"),
    "gemini-3-flash": ("models.gemini_flash", "GeminiFlash"),
    "qwen3-vl": ("models.qwen3_vl", "Qwen3VL"),
    "internvl3.5": ("models.internvl35", "InternVL35"),
    "qwen2-vl": ("models.qwen2_vl", "Qwen2VL"),
    "llava-1.5": ("models.llava15", "Llava15"),
}


def load_model_class(model_key: str):
    import importlib
    module_path, class_name = MODEL_REGISTRY[model_key]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

SETTINGS = ["T", "T+I", "T+E", "T+I+E"]

_HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(_HERE, "prompts")


def load_template(setting: str) -> str:
    fname = "fact_check_no_evi.md" if setting in ("T", "T+I") else "fact_check_evi.md"
    with open(os.path.join(PROMPT_DIR, fname), encoding="utf-8") as f:
        return f.read()


def format_evidence(evidence_list):
    if not evidence_list:
        return "(none provided)"
    return "\n".join(f"[{e['id']}] {e['content']}" for e in evidence_list)


def sample_evidence(evidence_list, k, seed, claim_id):
    """For the evidence-quantity ablation: deterministically pick at most k
    snippets out of evidence_list using a per-claim random permutation, so
    that the k=1 subset is contained in the k=2 subset, etc. (a clean
    dose-response curve). k=None or k >= len(evidence_list) means "use
    everything" (also covers claims that simply don't have k snippets)."""
    if k is None or not evidence_list or k >= len(evidence_list):
        return evidence_list
    order = evidence_list[:]
    random.Random(seed + claim_id).shuffle(order)
    return order[:k]


def build_prompt(template: str, claim: str, post_text: str, setting: str, evidence_list) -> str:
    # Plain string substitution (not str.format()) because the templates'
    # OUTPUT JSON schema block itself contains literal { } characters that
    # str.format() would otherwise try to interpret as fields.
    prompt = template.replace("{claim}", claim or "").replace("{post_context}", post_text or "")
    if setting in ("T", "T+I"):
        image_note = "(no image provided)" if setting == "T" else "(see attached image)"
        prompt = prompt.replace("{image_if_any}", image_note)
    else:
        prompt = prompt.replace("{evidence_items}", format_evidence(evidence_list))
    return prompt


def parse_model_json(raw_text: str):
    """Lightweight JSON extraction: direct parse, then strip markdown code
    fences, then fall back to the first {...} substring. No LLM-based
    repair call — keeps baseline evaluation cheap and simple."""
    if raw_text is None:
        return False, None
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    # Some local models (e.g. LLaVA-1.5) emit markdown-style escaped
    # underscores (\_) in field names, which is not a valid JSON escape
    # and would otherwise make an otherwise-correct response unparseable.
    text = text.replace("\\_", "_")

    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return True, json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return False, None


def build_output_record(claim_id, claim, setting, model_name, parsed,
                         evidence_total=None, evidence_shown=None):
    record = {
        "claim_id": claim_id,
        "post_info": {
            "claim": claim.get("claim"),
            "post_text": claim.get("post_text"),
            "post_image": claim.get("post_image"),
            "news_url": claim.get("news_url"),
            "merged_label": claim.get("merged_label"),
        },
        "setting": setting,
        "model": model_name,
        "final_answer": {
            "updated_answer": {
                "my_understanding_of_claim": parsed.get("my_understanding_of_claim"),
                "validation_result": {
                    "3-class_authenticity_label": parsed.get("label_3class"),
                    "evidence_ids": parsed.get("evidence_ids", []),
                    "reasoning_logic": parsed.get("reasoning_logic"),
                },
                "confidence_level": parsed.get("confidence_level"),
            }
        },
    }
    if evidence_total is not None:
        # Only meaningful for the evidence-quantity ablation; lets analysis
        # group by shown/total ratio as well as by raw shown count.
        record["evidence_meta"] = {
            "total_count": evidence_total,
            "shown_count": evidence_shown,
        }
    return record


def main():
    parser = argparse.ArgumentParser(description="Run T/T+E/T+I+E baseline LVLM evaluation")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--input_file", default=os.path.join(_PROJECT_ROOT, "Dataset_RW-Post", "test.jsonl"))
    parser.add_argument("--dataset", default="rwpost")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1, help="-1 = process all remaining claims")
    parser.add_argument("--output_root", default=os.path.join(_HERE, "results"))
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--evidence_k", type=int, default=None,
                         help="Evidence-quantity ablation: randomly sample at most K evidence "
                              "snippets per claim (fixed seed) instead of using all available "
                              "evidence. Ignored for setting=T. Omit for the full-evidence baseline.")
    parser.add_argument("--evidence_seed", type=int, default=42)
    args = parser.parse_args()

    model_cls = load_model_class(args.model)
    model = model_cls()
    template = load_template(args.setting)

    claims = extract_data_from_GT_jsonl(args.input_file, args.dataset)
    claim_slice = claims[args.start_id:] if args.end_id == -1 else claims[args.start_id:args.end_id]

    setting_dir = args.setting.replace("+", "")  # T, TE, TIE -> filesystem-safe
    if args.evidence_k is not None:
        setting_dir += f"_k{args.evidence_k}"
    output_dir = os.path.join(args.output_root, args.model, setting_dir)
    readable_dir = os.path.join(output_dir, "readable_json")
    os.makedirs(readable_dir, exist_ok=True)
    output_jsonl = os.path.join(output_dir, "output.jsonl")
    failed_jsonl = os.path.join(output_dir, "output_failed.jsonl")

    processed_urls = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    processed_urls.add(json.loads(line)["post_info"]["news_url"])
                except Exception:
                    continue
    print(f"Model={args.model} setting={args.setting} | already processed: {len(processed_urls)}")

    with open(output_jsonl, "a", encoding="utf-8") as out_f, \
         open(failed_jsonl, "a", encoding="utf-8") as fail_f:

        for offset, claim in enumerate(claim_slice):
            claim_id = args.start_id + offset
            news_url = claim.get("news_url")

            if news_url in processed_urls:
                print(f"[{claim_id}] skip (already processed)")
                continue

            full_evidence_list = claim.get("retrieved_evidence") or []
            evidence_list = sample_evidence(full_evidence_list, args.evidence_k, args.evidence_seed, claim_id) \
                if args.setting != "T" else full_evidence_list
            image_path = claim.get("post_image") if args.setting in ("T+I", "T+I+E") else None

            prompt = build_prompt(template, claim.get("claim"), claim.get("post_text"), args.setting, evidence_list)

            parsed = None
            last_raw = None
            last_error = None
            for attempt in range(args.max_retries):
                try:
                    raw = model.generate(prompt, image_path=image_path)
                    last_raw = raw
                    ok, data = parse_model_json(raw)
                    if ok and "label_3class" in data:
                        parsed = data
                        break
                    print(f"[{claim_id}] parse failed (attempt {attempt + 1}/{args.max_retries})")
                except Exception as e:
                    last_error = str(e)
                    print(f"[{claim_id}] model call failed (attempt {attempt + 1}/{args.max_retries}): {e}")

            if parsed is None:
                print(f"[{claim_id}] FAILED after {args.max_retries} attempts")
                fail_f.write(json.dumps({
                    "claim_id": claim_id,
                    "news_url": news_url,
                    "raw_output": last_raw,
                    "error": last_error,
                }, ensure_ascii=False) + "\n")
                fail_f.flush()
                continue

            record = build_output_record(claim_id, claim, args.setting, args.model, parsed,
                                          evidence_total=len(full_evidence_list),
                                          evidence_shown=len(evidence_list))
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            with open(os.path.join(readable_dir, f"{claim_id}.json"), "w", encoding="utf-8") as rf:
                json.dump(record, rf, ensure_ascii=False, indent=2)

            print(f"[{claim_id}] OK label={parsed.get('label_3class')} (gt={claim.get('merged_label')})")


if __name__ == "__main__":
    main()
