import os
import sys

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录（假设当前文件在子目录）
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# 加入 Python 模块搜索路径
sys.path.insert(0, project_root)

from datetime import datetime, timedelta
import os
import re
import json
import argparse
import copy
# from DIRE.demo_rewrite import predict_synthetic_probability
from retrieval_imagehash_with_google_vision import get_evidence, visual_search
from dataload import  extract_data_from_GT_jsonl
from agent_model import  call_Strategy_generator, call_Query_Site_generator, call_Middle_Reasoner, call_image_similarity_type_and_manipulation_judger, call_image_miscaption_detector, call_evidence_source_with_link_judger_detailed, call_Middle_Reasoner_with_Source_Judgment, call_Explainer_3_class_aligned
from agent_utils import extract_link, safe_parse_updated_answer, extract_never_used_ids

######################control table####################
parser = argparse.ArgumentParser(description="Run reasoning/evidence/image search batches with variable start_id")
parser.add_argument("--start_id", type=int, required=False, default=0, help="Starting index for batch processing")
parser.add_argument("--end_id", type=int, required=False, default=-1, help="Ending index for batch processing")
parser.add_argument("--input_file", type=str, required=False, default="Dataset_RW-Post/demo/demo.jsonl", help="the path to input jsonl")
parser.add_argument("--dataset", type=str, required=False, default="rwpost", help="the name of input dataset")
args = parser.parse_args()
start_id = args.start_id
end_id = args.end_id
input_jsonl = args.input_file
dataset = args.dataset

model_version = 'gpt-4o-mini'
default_agents_number = 1 
default_rounds_number = 1


strategy_use_flag = True
image_analysis_use_flag = True
image_evidence_use_flag = False


mode_name="dev"
search_mode = "open_book"  # "open_web" or "close_book" or "open_book"

sample_batch = start_id
output_dir = f'output/{dataset}/{mode_name}/{search_mode}/{str(sample_batch)}/{model_version}/'
os.makedirs(output_dir, exist_ok=True)
output_jsonl=output_dir+f'output.jsonl'
#用于标注执行模式的标记



#======================main process=========================
def main():
    """
    Main entry of fact-checking pipeline.

    Supports:
    - Resuming from processed samples
    - Processing a slice of claims [start_id, end_id)
    - If end_id == -1, process all claims after start_id
    """

    # ----------------------------
    # Output paths
    # ----------------------------
    fail_jsonl = output_dir + 'output_failed.jsonl'

    # ----------------------------
    # Load all claims from GT file
    # ----------------------------
    claims = extract_data_from_GT_jsonl(input_jsonl, dataset)

    # ----------------------------
    # Determine processing range
    # ----------------------------
    if end_id == -1:
        claim_slice = claims[start_id:]
        start_index = start_id
    else:
        claim_slice = claims[start_id:end_id]
        start_index = start_id

    # ----------------------------
    # Load already processed URLs
    # ----------------------------
    processed_urls = []
    try:
        with open(output_jsonl, 'r', encoding='utf-8') as pro_file:
            for line in pro_file:
                try:
                    data = json.loads(line)
                    processed_urls.append(data['post_info']['news_url'])
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
                except KeyError:
                    print("Skipping line due to missing post_info/news_url")
    except FileNotFoundError:
        print("Output file not found. Start from scratch.")
    except Exception as e:
        print(f"Error loading processed urls: {e}")

    print("Already processed URLs:", len(processed_urls))

    # ----------------------------
    # Open output files
    # ----------------------------
    error_count = 0

    with open(output_jsonl, 'a', encoding='utf-8') as outfile, \
         open(fail_jsonl, 'a', encoding='utf-8') as failfile:

        # enumerate 时保持 claim_id = 原始 claims 中的 index
        for offset, claim in enumerate(claim_slice):
            claim_id = start_index + offset

            # ----------------------------
            # Initialize per-claim contexts
            # ----------------------------
            context_reasoner = {}          # 中间推理（部分证据）
            context_final_reasoner = {}    # 最终推理（全部证据）
            context_query = {}             # 历史 query，防止重复生成
            context_record = {}            # 全量记录（写文件）
            context_imganalyzer = {}       # 图像证据推理结果

            tokens_amount_sum = 0
            step_idx = 0

            # ----------------------------
            # Parse post basic info
            # ----------------------------
            post = claim
            post_url = post.get('news_url')
            # post_image is a single path string, not a list (see dataload.py)
            post_imgs = post.get('post_image', '')
            claim_time = post.get('claim_time')

            print(f"\nProcessing claim_id={claim_id}, url={post_url}")

            # 跳过已经处理过的样本
            if post_url in processed_urls:
                print("Skip processed claim.")
                continue

            # ----------------------------
            # Parse claim_time and max search date
            # ----------------------------
            max_date = None
            if claim_time:
                try:
                    date_obj = datetime.strptime(
                        claim_time, "%Y-%m-%dT%H:%M:%SZ"
                    ).date()
                except ValueError:
                    try:
                        date_obj = datetime.strptime(
                            claim_time, "%Y-%m-%d"
                        ).date()
                    except ValueError:
                        print("Warning: invalid claim_time format:", claim_time)
                        date_obj = None

                if date_obj:
                    max_date = date_obj - timedelta(days=1)


            # ----------------------------
            # Step 1: Image search & analysis
            # ----------------------------
            image_Analysis_item = []
            
            if search_mode == "open_web" and post_imgs:
                image_search_results = []
                try:
                    image_search_results = visual_search(
                        post_imgs,
                        out_root=output_dir,
                        is_url=False
                    )
                except Exception as e:
                    print("Error during visual search:", e)

                if image_search_results:
                    context_reasoner[f'step_{step_idx}-image_search'] = \
                        json.dumps(image_search_results, ensure_ascii=False)
                    context_record[f'step_{step_idx}-image_search'] = \
                        json.dumps(image_search_results, ensure_ascii=False)
                    step_idx += 1


                # ----------------------------------------
                # Step 2: Image evidence analysis pipeline
                # ----------------------------------------

                #使用gpt分析图片证据
                # deepfake_analysis_flag = True
                # if deepfake_analysis_flag:
                #     img = post_imgs[0]
                #     output_deepfake_judger_score = predict_synthetic_probability(img)
                #     if output_deepfake_judger_score:
                #             context_imganalyzer[f'step_{i}-deepfake probability based on deepfake detection method']=str(output_deepfake_judger_score)
                #             context_reasoner[f'step_{i}-deepfake probability based on deepfake detection method']=str(output_deepfake_judger_score)
                #     else:
                #         print("no deepfake prediction result")
                #     i += 1

                img_page_extracted_list = image_search_results
                evidence_upgraded = []
                # Initialize image analyzer context
                if img_page_extracted_list:
                    context_imganalyzer['output_step_1'] = []  # source reliability & context
                    context_imganalyzer['output_step_2'] = []  # tampering analysis
                    context_imganalyzer['output_step_3'] = []  # miscaption analysis


                for page_result in img_page_extracted_list:
                    if not isinstance(page_result, dict):
                        continue
                    # ----------------------------------------
                    # Step 2.1: Source reliability judgment
                    # ----------------------------------------
                    output_step_1 = {}
                    source_link = page_result.get('href')

                    if source_link:
                        try:
                            output_step_1, total_tokens = \
                                call_evidence_source_with_link_judger_detailed(source_link)
                        except Exception:
                            output_step_1, total_tokens = \
                                call_evidence_source_with_link_judger_detailed(source_link)

                        tokens_amount_sum += total_tokens
                        output_step_1['updated_answer'] = safe_parse_updated_answer(
                            output_step_1,
                            caller_name="call_evidence_source_with_link_judger_detailed"
                        )
                        context_imganalyzer['output_step_1'].append(output_step_1)
                    else:
                        continue

                    # ----------------------------------------
                    # Step 2.2: Image similarity & tampering analysis
                    # ----------------------------------------
                    output_step_2 = {}

                    if page_result.get("img_address"):
                        try:
                            output_step_2, total_tokens = \
                                call_image_similarity_type_and_manipulation_judger(page_result, post)
                        except Exception:
                            output_step_2, total_tokens = \
                                call_image_similarity_type_and_manipulation_judger(page_result, post)

                        tokens_amount_sum += total_tokens
                        output_step_2['updated_answer'] = safe_parse_updated_answer(
                            output_step_2,
                            caller_name="call_image_similarity_type_and_manipulation_judger"
                        )
                        context_imganalyzer['output_step_2'].append(output_step_2)

                    # ----------------------------------------
                    # Step 2.3: Image miscaption probability analysis
                    # ----------------------------------------
                    output_step_3 = {}

                    if len(page_result.get("title", "")) > 1 or len(page_result.get("body", "")) > 1:
                        try:
                            output_step_3, total_tokens = \
                                call_image_miscaption_detector(page_result, post, "")
                        except Exception:
                            output_step_3, total_tokens = \
                                call_image_miscaption_detector(page_result, post, "")

                        tokens_amount_sum += total_tokens
                        output_step_3['updated_answer'] = safe_parse_updated_answer(
                            output_step_3,
                            caller_name="call_image_miscaption_detector"
                        )
                        context_imganalyzer['output_step_3'].append(output_step_3)

                    # ----------------------------------------
                    # Step 2.4: Aggregate upgraded evidence item
                    # ----------------------------------------
                    if isinstance(output_step_1.get('updated_answer'), dict):

                        evidence_key = f"evidence_{evidence_i}"
                        evidence_item = {
                            evidence_key: {
                                "source_judgment_result": output_step_1['updated_answer']
                            }
                        }

                        if isinstance(output_step_2.get('updated_answer'), dict):
                            evidence_item[evidence_key]["image_tampering_analysis_result"] = {
                                "tampering_probability":
                                    output_step_2['updated_answer'].get("tampering_probability", ""),
                                "tampering_reasoning":
                                    output_step_2['updated_answer'].get("tampering_reasoning", "")
                            }

                        if isinstance(output_step_3.get('updated_answer'), dict):
                            evidence_item[evidence_key]["image_miscaption_analysis_result"] = \
                                output_step_3['updated_answer']
                            evidence_item[evidence_key]["note"] = (
                                "Because the tampering and miscaption analysis assume the source "
                                "is reliable, the source_judgment_result should be considered "
                                "when drawing the final conclusion."
                            )

                        evidence_upgraded.append(evidence_item)
                        evidence_i += 1

                # ----------------------------------------
                # Preserve image analysis results
                # ----------------------------------------
                image_Analysis_item = evidence_upgraded

            # Accumulated reasoning content used as context for strategy generation
            content_reasoning = ""

            # Maximum number of iterative reasoning / search steps
            max_step = 10

            # ----------------------------------------
            # Iterative fact-checking loop
            # ----------------------------------------
            # This loop alternates between:
            # 1. Strategy generation
            # 2. Query generation
            # 3. (Next steps: evidence retrieval & reasoning)
            stop = False
            i = 0
            # Defaults so these remain defined even if the loop below breaks
            # on its first iteration (e.g. strategy generation fails).
            sim_validation_result = ""
            last_sim_validation_result = ""
            filtered_information = []
            context_final_reasoner["information list"] = []
            while not stop and i < max_step:
                # Indicator for whether collected information is sufficient
                information_sufficiency = 0

                # ==================================================
                # Step 3: Strategy generation
                # ==================================================
                try_strategy = 0
                failure_strategy = True

                # Retry strategy generation up to 2 times
                while failure_strategy and try_strategy < 2:
                    content_strategy, total_tokens = call_Strategy_generator(
                        post,
                        content_reasoning
                    )
                    tokens_amount_sum += total_tokens

                    # Strategy output must contain "updated_answer"
                    if 'updated_answer' not in content_strategy:
                        print("Strategy generation failed, retry:", try_strategy)
                        try_strategy += 1
                        continue

                    strategy = content_strategy['updated_answer']

                    # Store strategy in multiple contexts
                    context_reasoner['strategy'] = strategy
                    context_final_reasoner['strategy'] = strategy
                    context_record['strategy'] = strategy

                    # Strategy must include both search_list and validation_list
                    search_list = strategy.get('search_list')
                    validation_list = strategy.get('validation_list')

                    if search_list is None or validation_list is None:
                        print("Invalid strategy format, retry:", try_strategy)
                        try_strategy += 1
                        continue

                    # Strategy successfully generated
                    failure_strategy = False

                # If strategy generation fails completely, stop the reasoning loop
                if failure_strategy:
                    print("Strategy generation failed after retries. Stop reasoning loop.")
                    break

                # ==================================================
                # Step 4: Query generation
                # ==================================================
                # Final list of query dictionaries used for evidence retrieval
                query_values = []

                # Initialize information containers for this step
                context_reasoner["information list"] = []
                context_final_reasoner["information list"] = []

                context_record["information list"] = []
                context_record["information list with query"] = []

                # Evidence bookkeeping
                evidence_id = 0
                never_used_info_ids = []

                # Variables for similarity-based validation (used later)
                sim_validation_result = ""
                last_sim_validation_result = ""
                filtered_information = []
                if search_mode == "open_web":
                    # ----------------------------------------
                    # 4.1 Validation list → direct sentence queries
                    # ----------------------------------------
                    # Validation items already provide concrete sentences,
                    # so they are directly used as search queries.
                    if validation_list:
                        for validation_item in validation_list:
                            if 'sentence' in validation_item:
                                sentence = validation_item['sentence']
                                query_values.append({"query": [sentence]})

                                context_query[f'step_{i}-search query and evidence'] = {
                                    'queries': [sentence]
                                }

                    # ----------------------------------------
                    # 4.2 Search list → LLM-generated queries
                    # ----------------------------------------
                    # Search items require query generation via LLM
                    if search_list:
                        try:
                            content_A, total_tokens = call_Query_Site_generator(
                                search_list,
                                post
                            )
                        except Exception:
                            content_A, total_tokens = call_Query_Site_generator(
                                search_list,
                                post
                            )

                        tokens_amount_sum += total_tokens

                        generated_queries = content_A.get('updated_answer', [])

                        context_query[f'step_{i}-search query and evidence'] = {
                            "search_item": search_list,
                            "queries": generated_queries
                        }

                        # Flatten generated queries into query_values
                        for query_item in generated_queries:
                            if 'queries' in query_item:
                                for query in query_item['queries']:
                                    query_values.append({"query": [query]})

                    # ----------------------------------------
                    # Record generated queries for this step
                    # ----------------------------------------
                    context_record[f'step_{i}-search queries'] = query_values
                    i += 1

                    # ----------------------------------------
                    # Placeholder: next step will retrieve text evidence
                    # ----------------------------------------
                    retrieved_text = None


                    # ==================================================
                    # Step 4: Evidence retrieval and preliminary reasoning
                    # ==================================================
                    # Iterate through all generated query values and retrieve evidence.
                    # After each retrieval, reasoning may be performed to decide
                    # whether further evidence is still needed.
                    for query_index, query_web_list in enumerate(query_values):

                        # Stop immediately if global stop flag is set
                        if stop:
                            break

                        # ----------------------------------------
                        # Determine target websites (if specified)
                        # ----------------------------------------
                        if "websites" in query_web_list:
                            sites = query_web_list["websites"]
                        elif "recommended_sources" in query_web_list:
                            sites = query_web_list["recommended_sources"]
                        else:
                            print("No recommended websites provided.")
                            sites = ""

                        # Extract search queries
                        queries = query_web_list["query"]

                        # ----------------------------------------
                        # Evidence retrieval retry loop
                        # ----------------------------------------
                        try_retrieve = 0
                        retrieved_text_empty_flag = True
                        # Defaults so these remain defined even if get_evidence()
                        # raises on every attempt below.
                        retrieved_list = []
                        retrieved_list_with_link = []
                        retrieved_text_with_link = ""

                        # Retry retrieval if no evidence is found
                        while (
                            retrieved_text_empty_flag
                            and try_retrieve < 1
                            and not stop
                            and i < max_step
                        ):
                            try:
                                # Retrieve evidence using search engine API
                                retrieved_list, retrieved_list_with_link, \
                                retrieved_text, retrieved_text_with_link = get_evidence(
                                    post['post_text'],
                                    post['claim'],
                                    queries,
                                    out_root=output_dir,
                                    max_date=max_date
                                )
                            except Exception as e:
                                print("Error occurred during evidence retrieval:", e)
                                try_retrieve += 1
                                retrieved_text = ""
                                continue

                            # Record retrieved evidence (full version with links) for logging
                            context_record[f'step_{i}-search query and evidence'] = {
                                'querys': queries,
                                'retrieved_list_with_link': retrieved_list_with_link
                            }
                            i += 1

                            # ----------------------------------------
                            # Ensure retrieved_text is a dictionary
                            # ----------------------------------------
                            if isinstance(retrieved_text, str):
                                try:
                                    retrieved_text = json.loads(retrieved_text)
                                    print("retrieved_text converted from string to dict.")
                                except json.JSONDecodeError as e:
                                    print("Failed to parse retrieved_text JSON:", e)
                                    print("retrieved_text:", retrieved_text)
                                    try_retrieve += 1
                                    continue

                            # ----------------------------------------
                            # Check whether any evidence content is non-empty
                            # ----------------------------------------
                            for key, value in retrieved_text.items():
                                if value == "[]" or value == []:
                                    print(f"Evidence under '{key}' is empty.")
                                else:
                                    print(f"Evidence under '{key}' is not empty.")
                                    retrieved_text_empty_flag = False

                            # ----------------------------------------
                            # If no evidence found, record empty placeholder
                            # ----------------------------------------
                            if retrieved_text_empty_flag:
                                try_retrieve += 1
                                context_record["information list with query"].append({
                                    "id": "",
                                    "content": "",
                                    "query": queries
                                })

                        # ----------------------------------------
                        # Append retrieved evidence to reasoning context
                        # ----------------------------------------
                        # Each piece of evidence is assigned a unique evidence_id
                        for text_plain, text_with_link in zip(
                            retrieved_list,
                            retrieved_list_with_link
                        ):
                            context_reasoner["information list"].append({
                                "id": evidence_id,
                                "content": text_with_link
                            })
                            context_record["information list"].append({
                                "id": evidence_id,
                                "content": text_with_link
                            })
                            context_record["information list with query"].append({
                                "id": evidence_id,
                                "content": text_with_link,
                                "query": query_web_list
                            })
                            evidence_id += 1





                    # ==================================================
                    # Dual-reasoning mode
                    # ==================================================
                    # To reduce reasoning cost, reasoning is triggered only when:
                    # 1. At least 2 pieces of evidence are available, OR
                    # 2. This is the last query, OR
                    # 3. The step limit is about to be reached
                    if (
                        (context_reasoner["information list"] and len(context_reasoner["information list"]) > 1)
                        or query_index > len(query_values) - 2
                        or i > (max_step - 1)
                    ):
                        # Preserve the last reasoning result
                        last_sim_validation_result = sim_validation_result

                        # Backup current reasoning context
                        context_reasoner_remain = copy.deepcopy(context_reasoner)

                        # ----------------------------------------
                        # Split evidence into batches if too many
                        # ----------------------------------------
                        # If more than 10 evidence items exist, reason on the first batch
                        # and postpone the rest to the next reasoning round
                        if len(context_reasoner["information list"]) > 10:
                            context_reasoner_remain["information list"] = \
                                context_reasoner_remain["information list"][10:]
                            context_reasoner["information list"] = \
                                context_reasoner["information list"][:10]

                            print(
                                f"{len(context_reasoner_remain['information list'])} evidence items "
                                "will be postponed to the next reasoning round."
                            )
                        else:
                            context_reasoner_remain["information list"] = []

                        # Debug print: list evidence used in current batch
                        for idx, item in enumerate(context_reasoner["information list"]):
                            print(idx, item)

                        # ==================================================
                        # Reasoning Phase 1:
                        # Batch-level preliminary reasoning
                        # ==================================================
                        # Purpose:
                        # - Filter out irrelevant evidence
                        # - Produce preliminary validation result
                        try:
                            content_reasoning, total_tokens = call_Middle_Reasoner(
                                image_Analysis_item,
                                context_reasoner,
                                post
                            )
                        except Exception:
                            content_reasoning, total_tokens = call_Middle_Reasoner(
                                image_Analysis_item,
                                context_reasoner,
                                post
                            )

                        tokens_amount_sum += total_tokens

                        # Extract reasoning result
                        sim_validation_result = content_reasoning.get('updated_answer', {})
                        context_record[f'step_{i}-middle_reasoner_result_sim'] = sim_validation_result
                        i += 1

                        # ----------------------------------------
                        # Filter unused evidence based on reasoning
                        # ----------------------------------------
                        never_used_info_ids = extract_never_used_ids(
                            sim_validation_result,
                            context_reasoner["information list"]
                        )

                        filtered_information = [
                            info for info in context_reasoner.get("information list", [])
                            if info.get("id") not in never_used_info_ids
                        ]

                        # ==================================================
                        # Accumulate useful evidence for final reasoning
                        # ==================================================
                        length_before_update = len(context_final_reasoner["information list"])

                        for info in filtered_information:
                            source_link = extract_link(info["content"])
                            source_assessment_result, total_tokens = \
                                call_evidence_source_with_link_judger_detailed(source_link)

                            tokens_amount_sum += total_tokens
                            info["source_judgment_result"] = source_assessment_result
                            context_final_reasoner["information list"].append(info)

                        # Restore postponed evidence for next round
                        context_reasoner["information list"] = \
                            context_reasoner_remain["information list"]

                        # ----------------------------------------
                        # Check sufficiency after batch reasoning
                        # ----------------------------------------
                        information_sufficiency = (
                            sim_validation_result
                            .get("validation_result", {})
                            .get("final_sufficiency_confidence", "")
                        )

                        predicted_label = (
                            sim_validation_result
                            .get("validation_result", {})
                            .get("overall_authenticity", "")
                        )

                        print("***********************************************************")
                        print(
                            "Current accumulated evidence count:",
                            len(context_final_reasoner["information list"])
                        )
                        print("***********************************************************")

                        # Early stop:
                        # If this is the first effective reasoning and sufficiency is high
                        if (
                            length_before_update == 0
                            and information_sufficiency
                            and int(information_sufficiency) > 4
                            and 'unproven' not in predicted_label.lower()
                        ):
                            stop = True
                            break

                        # ==================================================
                        # Reasoning Phase 2:
                        # Accumulated evidence reasoning
                        # ==================================================
                        # Triggered only if:
                        # - This is not the first batch
                        # - New useful evidence was found
                        if filtered_information and length_before_update > 0:

                            # If too many accumulated evidence items exist,
                            # force this to be the final reasoning step
                            if len(context_final_reasoner["information list"]) > 5:
                                print(
                                    "Evidence count exceeds 5:",
                                    len(context_final_reasoner["information list"])
                                )
                                stop = True

                            try:
                                content_reasoning, total_tokens = call_Middle_Reasoner(
                                    image_Analysis_item,
                                    context_final_reasoner,
                                    post,
                                    agents=default_agents_number,
                                    rounds=default_rounds_number
                                )
                            except Exception:
                                content_reasoning, total_tokens = call_Middle_Reasoner(
                                    image_Analysis_item,
                                    context_final_reasoner,
                                    post,
                                    agents=default_agents_number,
                                    rounds=default_rounds_number
                                )

                            tokens_amount_sum += total_tokens

                        # Record accumulated reasoning result
                        sim_validation_result = content_reasoning.get('updated_answer', {})
                        context_record[f'step_{i}-accumulated_reasoner_result_sim'] = \
                            sim_validation_result
                        i += 1

                        # ----------------------------------------
                        # Final sufficiency check
                        # ----------------------------------------
                        information_sufficiency = (
                            sim_validation_result
                            .get("validation_result", {})
                            .get("final_sufficiency_confidence", "")
                        )

                        print("***********************************************************")
                        print(
                            "Current accumulated evidence count:",
                            len(context_final_reasoner["information list"])
                        )
                        print("***********************************************************")

                        # Stop if sufficiency requirement is satisfied
                        if (
                            information_sufficiency
                            and int(information_sufficiency) > 3
                            and 'unproven' not in predicted_label.lower()
                        ):
                            stop = True
                            break

                        # Hard stop if maximum step limit is reached
                        if i > (max_step - 1):
                            stop = True




                if search_mode == "open_book":
                    context_record["information list"] = claim.get("retrieved_evidence", [])
                    stop = True  # In close-book mode, reasoning is done in one step without iterative search
                if search_mode == "close_book":
                    context_record["information list"] = []
                    stop = True  # In close-book mode, reasoning is done in one step without iterative search

            # ==================================================
            # Final cleanup after all search and reasoning steps
            # ==================================================

            # Remove evidence items that were never used during reasoning
            # and keep only evidence with valid links
            if context_record.get("information list", []):

                context_record["information list with link"] = [
                    info
                    for info in context_record.get("information list", [])
                    if info.get("id") not in never_used_info_ids
                ]

                # Synchronize final reasoner evidence list
                context_final_reasoner["information list"] = \
                    context_record["information list with link"]

            # ==================================================
            # Fallback logic if reasoning loop did not stop normally
            # ==================================================
            # This situation occurs when:
            # - The reasoning loop terminated due to step limit
            # - No final reasoning was triggered using all accumulated evidence
            if not stop:

                # If no new useful evidence was found in the last batch,
                # but a previous reasoning result exists,
                # fall back to the most recent valid reasoning output
                if (
                    len(filtered_information) == 0
                    and len(str(last_sim_validation_result)) > 0
                ):
                    sim_validation_result = last_sim_validation_result

            # ==================================================
            # Final reasoning with source judgment
            # ==================================================

            failed = True
            retry_reasoner = 0

            # ----------------------------------------
            # Final accumulated reasoning with source judgment
            # ----------------------------------------
            content_reasoning, total_tokens = call_Middle_Reasoner_with_Source_Judgment(
                image_Analysis_item,
                context_final_reasoner,
                post
            )

            tokens_amount_sum += total_tokens

            sim_validation_result = content_reasoning.get('updated_answer', {})
            context_record[f'step_{i}-accumulated_reasoner_result_sim'] = sim_validation_result
            i += 1


            # ==================================================
            # Final explanation generation (retryable)
            # ==================================================
            while failed and retry_reasoner < 2:

                try:
                    content, total_tokens = call_Explainer_3_class_aligned(
                        image_Analysis_item,
                        {'information list': context_final_reasoner['information list']},
                        sim_validation_result,
                        post
                    )
                except Exception:
                    content, total_tokens = call_Explainer_3_class_aligned(
                        image_Analysis_item,
                        {'information list': context_final_reasoner['information list']},
                        sim_validation_result,
                        post
                    )

                tokens_amount_sum += total_tokens

                # ----------------------------------------
                # Validate explainer output format
                # ----------------------------------------
                if not content or 'updated_answer' not in content:
                    print("Error: explainer returned invalid output.")
                    print("Raw content:", str(content))
                    retry_reasoner += 1
                    continue

                updated_answer = content['updated_answer']
                updated_answer_str = str(updated_answer).lower()

                # ----------------------------------------
                # Extract confidence level
                # ----------------------------------------
                confidence_match = re.search(
                    r"'confidence_level':\s*'?\b(\d+)\b'?",
                    updated_answer_str
                )

                if not confidence_match:
                    print("Error: failed to extract confidence level.")
                    print("updated_answer:", updated_answer_str)
                    retry_reasoner += 1
                    continue

                confidence_level = confidence_match.group(1)

                # ----------------------------------------
                # Extract 3-class authenticity label
                # ----------------------------------------
                authenticity_matches = re.findall(
                    r"'3-class_authenticity_label': '([^']+)'",
                    updated_answer_str
                )

                if not authenticity_matches:
                    print("Error: failed to extract authenticity label.")
                    print("updated_answer:", updated_answer_str)
                    retry_reasoner += 1
                    continue

                authenticity_label = authenticity_matches[0].lower()

                # ----------------------------------------
                # Build final structured output
                # ----------------------------------------
                final_answer = {
                    'updated_answer': {
                        'validation_result': {
                            '2-class_authenticity_label': (
                                updated_answer
                                .get('validation_result', {})
                                .get('2-class_authenticity_label')
                            ),
                            '3-class_authenticity_label': authenticity_label,
                            'reasoning_logic': (
                                updated_answer
                                .get('validation_result', {})
                                .get('reasoning_logic')
                            ),
                            'key_points': (
                                updated_answer
                                .get('validation_result', {})
                                .get('key_points', [])
                            )
                        },
                        'motivation': updated_answer.get('motivation', {}),
                        'confidence_level': confidence_level,
                        'evidence': updated_answer.get('evidence', [])
                    },
                    'influence_type': content.get('influence_type'),
                    'time_seconds': content.get('time_seconds'),
                    'tokens_cost': tokens_amount_sum
                }

                failed = False

            # If the explainer still failed after retries, there is no
            # final_answer to persist. Record the claim as failed instead of
            # crashing the whole batch, and move on to the next claim.
            if failed:
                print(f"Explainer failed after retries for claim_id={claim_id}. Recording as failed.")
                error_count += 1
                try:
                    json.dump(post, failfile)
                    failfile.write('\n')
                except Exception:
                    print(f"Error saving failed claim, error_count={error_count}")
                continue

            # ==================================================
            # Persist final output
            # ==================================================
            new_row = {
                "claim_id": claim_id,
                "post_info": post,
                "context": context_record,
                "final_answer": final_answer
            }

            try:
                # Append JSONL output
                json.dump(new_row, outfile)
                outfile.write('\n')

                print("Saved row:", new_row)

                # Save readable pretty JSON
                readable_path = os.path.join(
                    output_dir,
                    "readable_json",
                    f"{claim_id}.json"
                )
                os.makedirs(os.path.dirname(readable_path), exist_ok=True)

                with open(readable_path, 'w', encoding='utf-8') as pretty_file:
                    json.dump(new_row, pretty_file, indent=4, ensure_ascii=False)

                print(f"Saved claim {claim_id}")

            except Exception:
                print(f"Error saving final answer, error_count={error_count}")
                error_count += 1
                json.dump(post, failfile)

main()
