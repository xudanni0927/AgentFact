import os
import cv2

import json
import re
from typing import Any, Dict, Optional, Tuple, Union, List

Key = str
JSONLike = Union[dict, list, str, int, float, bool, None]

# 正则：匹配 step_<数字> 与中间连接符及空格的各种写法

PATTERN_IMAGE_SEARCH = re.compile(
    r"^step_(\d+)[\s\-_]*(?:image[\s\-_]*search[\s\-_]*result|evidence[\s\-_]*with[\s\-_]*similar[\s\-_]*image)$",
    re.IGNORECASE
)

PATTERN_SIMILAR_EVID  = re.compile(r"^step_(\d+)[\s\-_]*similar[\s\-_]*image[\s\-_]*evidence$", re.IGNORECASE)

def _maybe_parse_json(value: JSONLike) -> JSONLike:
    """若 value 是 JSON 字符串则尝试解析，否则原样返回。解析失败也原样返回。"""
    if isinstance(value, str):
        v = value.strip()
        if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
            try:
                return json.loads(v)
            except Exception:
                return value
    return value




#extract input from collected news' GT-steps file and remove the image damaged json
def extract_data_from_GT_jsonl(file_path, dataset, with_image_mode=True, with_evidence_mode=True):
    extracted_data = []
    filtered_file_path = file_path.replace(".jsonl", "_imagefiltered.jsonl")

    with open(file_path, 'r', encoding='utf-8') as infile, \
         open(filtered_file_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            try:
                # Parse each JSON line
                data = json.loads(line.strip())

                # Extract necessary fields
                claim = data.get("claim", "")
                news_url = data.get("news_url", "")
                post_info = data.get("steps_in_groundtruth_generated", {}).get("Post", {})
                post_text = post_info.get("text", "")

                extracted_item = {
                    "claim": claim,
                    "post_text": post_text,
                    "post_image": None,
                    "claim_time": data.get("claim_time"),
                    "news_url": news_url,
                    "merged_label": data.get("label", ""),
                    "fake_type": data.get("sub_label", ""),
                    "retrieved_evidence": None,
                }
                # 前缀映射表：旧前缀 -> 新前缀
                prefix_map = {
                    "./": "Dataset_RW-Post/"
                }

                def replace_prefix(path):
                    for old, new in prefix_map.items():
                        if path.startswith(old):
                            return path.replace(old, new, 1)
                    return path

                post_images = [replace_prefix(img) for img in post_info.get("image_address", [])]

                if dataset == "mocheg" or dataset == "claimreview_noimg" :
                    with_image_mode = False
                if with_image_mode and post_images:
                    image = cv2.imread(post_images[0])
                    if image is not None:
                        # Keep valid data and write it to the new file
                        extracted_item["post_image"] = post_images[0]
                        extracted_data.append(extracted_item)
                    else:
                        print("Image cannot be correctly opened:", post_images[0])
                        continue
                else:
                    print("No images found for sample")
                    if not with_image_mode:
                        extracted_data.append(extracted_item)

                if with_evidence_mode:
                    try:
                        evidence_list = (
                            data.get("steps_in_groundtruth_generated", {})
                                .get("Sub_step_5", {})
                                .get("fact_checking_evidence", [])
                        )

                        if isinstance(evidence_list, list):
                            lines = []
                            for e in evidence_list:
                                if not isinstance(e, dict):
                                    continue

                                evidence_id = e.get("evidence_id", "unknown")
                                evidence_content = e.get("evidence_content", "")

                                if evidence_content:
                                    lines.append({"id": evidence_id, "content": evidence_content})

                            extracted_item["retrieved_evidence"] = lines
                    except Exception as e:
                        # 可根据需要替换为 logging
                        print(f"Warning: failed to parse evidence: {e}")
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"Error decoding JSON line or processing data: {line}")

    return extracted_data
