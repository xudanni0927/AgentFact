"""
Evidence-quantity ablation, ratio view.

Pools records from all K-tier runs (TE_k1, TE_k2, ..., TE_k9, TE=full) and
re-buckets every record by its OWN evidence coverage ratio
(shown_count / total_count for that claim), instead of by the nominal K
used to generate it. This normalizes away the fact that different claims
have different total amounts of ground-truth evidence (mean 5.36, range
0-17), answering "does accuracy track the fraction of evidence shown"
rather than "does accuracy track the raw count".

Usage:
    python analyze_evidence_ratio.py
"""
import os
import json
from sklearn.metrics import classification_report

_HERE = os.path.dirname(os.path.abspath(__file__))
K_TIERS = ["TE_k1", "TE_k2", "TE_k3", "TE_k5", "TE_k7", "TE_k9", "TE"]

label_mapping = {
    "refuted": "false", "supported": "true", "nei": "unproven",
    "true": "true", "false": "false", "unproven": "unproven",
}
valid_labels = {"true", "false", "unproven"}


def clean_and_map(label):
    return label_mapping.get(str(label).strip().lower(), "unknown")


def load_records():
    records = []
    for tier in K_TIERS:
        path = os.path.join(_HERE, "results", "gpt-4o-mini", tier, "output.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                post_info = obj.get("post_info", {})
                validation = obj["final_answer"]["updated_answer"].get("validation_result", {})
                meta = obj.get("evidence_meta") or {}
                total = meta.get("total_count")
                shown = meta.get("shown_count")

                gt = clean_and_map(post_info.get("merged_label", ""))
                pred = clean_and_map(validation.get("3-class_authenticity_label", ""))
                if gt not in valid_labels or pred not in valid_labels:
                    continue
                if not total:  # total_count is 0 or None -> ratio undefined
                    continue

                records.append({
                    "claim_id": obj.get("claim_id"),
                    "tier": tier,
                    "total": total,
                    "shown": shown,
                    "ratio": shown / total,
                    "gt": gt,
                    "pred": pred,
                })
    return records


def bucket_label(ratio):
    if ratio >= 1.0:
        return "100%"
    if ratio >= 0.8:
        return "80-100%"
    if ratio >= 0.6:
        return "60-80%"
    if ratio >= 0.4:
        return "40-60%"
    if ratio >= 0.2:
        return "20-40%"
    return "0-20%"


BUCKET_ORDER = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "100%"]


def main():
    records = load_records()
    print(f"Pooled {len(records)} records across {len(K_TIERS)} K-tiers "
          f"(dedup note: same claim appears once per tier it ran in).\n")

    buckets = {b: [] for b in BUCKET_ORDER}
    for r in records:
        buckets[bucket_label(r["ratio"])].append(r)

    print(f"{'ratio bucket':<10} {'n':>6} {'3cls-Acc':>9} {'MacroF1':>8} {'WgtF1':>8} "
          f"{'bin-Acc':>8} {'bin-F1':>8}")
    for b in BUCKET_ORDER:
        rows = buckets[b]
        if not rows:
            print(f"{b:<10} {'(empty)':>6}")
            continue
        y_true = [r["gt"] for r in rows]
        y_pred = [r["pred"] for r in rows]
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        bin_true = ["true" if x == "true" else "not_true" for x in y_true]
        bin_pred = ["true" if x == "true" else "not_true" for x in y_pred]
        bin_rep = classification_report(bin_true, bin_pred, output_dict=True, zero_division=0)
        print(f"{b:<10} {len(rows):>6} {rep['accuracy']:>9.4f} "
              f"{rep['macro avg']['f1-score']:>8.4f} {rep['weighted avg']['f1-score']:>8.4f} "
              f"{bin_rep['accuracy']:>8.4f} {bin_rep['weighted avg']['f1-score']:>8.4f}")

    out_path = os.path.join(_HERE, "results", "gpt-4o-mini", "evidence_ratio_analysis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Pooled {len(records)} records across {len(K_TIERS)} K-tiers\n\n")
        f.write(f"{'ratio bucket':<10} {'n':>6} {'3cls-Acc':>9} {'MacroF1':>8} {'WgtF1':>8} "
                f"{'bin-Acc':>8} {'bin-F1':>8}\n")
        for b in BUCKET_ORDER:
            rows = buckets[b]
            if not rows:
                f.write(f"{b:<10} {'(empty)':>6}\n")
                continue
            y_true = [r["gt"] for r in rows]
            y_pred = [r["pred"] for r in rows]
            rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            bin_true = ["true" if x == "true" else "not_true" for x in y_true]
            bin_pred = ["true" if x == "true" else "not_true" for x in y_pred]
            bin_rep = classification_report(bin_true, bin_pred, output_dict=True, zero_division=0)
            f.write(f"{b:<10} {len(rows):>6} {rep['accuracy']:>9.4f} "
                    f"{rep['macro avg']['f1-score']:>8.4f} {rep['weighted avg']['f1-score']:>8.4f} "
                    f"{bin_rep['accuracy']:>8.4f} {bin_rep['weighted avg']['f1-score']:>8.4f}\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
