from agent_utils import llm_process, image_to_code

# =====================================================================
# [Paper] This file implements the five AgentFact agents described in
# the Method section (Sec. III) of the paper. Each function below is
# annotated with which paper agent it corresponds to. Overview:
#
#   Agent-SP (Strategy Planning)              -> call_Strategy_generator
#   Agent-TR (Text Retrieval & Validation)     -> call_Query_Site_generator,
#                                                 call_evidence_source_with_link_judger_detailed
#   Agent-IR (Image Retrieval & Analysis)      -> call_image_similarity_type_and_manipulation_judger,
#                                                 call_image_miscaption_detector
#   Agent-R  (Reasoning)                       -> call_Middle_Reasoner,
#                                                 call_Middle_Reasoner_with_Source_Judgment
#   Agent-EG (Explanation Generation)          -> call_Explainer_3_class_aligned
#
# call_ImageRelevancyJudger, source_analyzer, and call_evidence_source_judger_detailed
# are earlier/alternate implementations that are NOT wired into
# main_workflow.py (not imported/called there) — kept for reference only.
#
# 本文件实现了论文方法章节（第III节）里描述的五个 AgentFact 智能体。下面每个
# 函数上方都标注了它对应论文里的哪个 agent。总览：
#
#   Agent-SP（策略规划）         -> call_Strategy_generator
#   Agent-TR（文本证据检索验证） -> call_Query_Site_generator,
#                                  call_evidence_source_with_link_judger_detailed
#   Agent-IR（图像检索分析）     -> call_image_similarity_type_and_manipulation_judger,
#                                  call_image_miscaption_detector
#   Agent-R（推理）              -> call_Middle_Reasoner,
#                                  call_Middle_Reasoner_with_Source_Judgment
#   Agent-EG（解释生成）         -> call_Explainer_3_class_aligned
#
# call_ImageRelevancyJudger、source_analyzer、call_evidence_source_judger_detailed
# 是较早/备用的实现，并未被 main_workflow.py 引用调用（未 import/未调用），
# 仅作历史参考保留。
# =====================================================================

background = (
    "The fact-checking framework is designed to identify misinformation in social media posts."
    "You will receive the post's text, images, and main claim, along with context generated from previous steps, "
    "including gathered evidence, and previous fact-checking analysis result. " 
)

# ---------------------------------------------------------------------
# [Paper] Agent-SP — Strategy Planning Agent (Sec. III-C.1; Workflow Step 1).
# Generates the verification plan: reasoning steps (S), a validation list
# (L_v) of statements to check, and a search list (L_s) of search intents.
# 论文对应：策略规划智能体 Agent-SP（方法节 III-C.1；工作流第1步）。
# 生成核查计划：推理步骤(S)、待验证陈述清单(L_v)、辅助搜索意图清单(L_s)。
# ---------------------------------------------------------------------
def call_Strategy_generator(post, content_reasoning):
    print("*Step* Strategy_generator")
    prompt_role = "You are an fact-checking plan designer for a fact-checking framework."
    prompt_task = """Based on the current context, design or refine a validation plan and an information search/validation list to efficiently and accurately analyze and validate the post claim. 
    ### **Fact-Checking Techniques**
    Depending on the nature of the claim, apply one or more of the following fact-checking techniques:
    - **Divide and Conquer**: Break complex claims into atomic sub-parts for separate verification.
    - **Origin Tracing & Information Evolution**: Identify the earliest appearance of the media or claim and how the narrative has evolved across platforms or time.
    - **Chain of Evidence**: Check logical continuity between components of the story (e.g., date, place, photo, action).
    - **Cross-Verification**: Confirm with multiple, independent, and preferably primary sources.
    - **Primary Source Verification**: Trace information back to the first party or eyewitness sources.
    - **Temporal**: Ensure all elements of the claim align chronologically.
    - **Reverse Image Search & Metadata Analysis**: Determine when and where an image first appeared.
    - **Source Credibility Assessment** Consistency Check: Assess trustworthiness and independence of cited sources.
    - **Logical Consistency Analysis**: Spot internal contradictions in the story.

    ### **Your Response Should Include:**
    1. **Validation Logic**  
    - A structured and specific reasoning process that outlines how the claim should be analyzed.  
    - Indicate which fact-checking techniques are most relevant for this case.  

    2. **Validation List (Information to Verify)**  
    - A list of **original** sentences that require verification (which means we expect to use this sentence as 'queries' on search engine to validate the original source of this information.) (maximum **3 items**).  
    - Avoid redundant or trivial sentences.  
    - If there is no original sentence to validate, you can leave the item empty.

    3. **Information Search List (Key Search Queries)**  
    - A prioritized list (at most **3 items**) of key information directly required for fact-checking.  
    - **Ordered by priority**: from the **most important information** (e.g.,direct evidence) to the **more contextual or secondary sources** (e.g., expert opinions).  
    - **Avoid redundancy**: Ensure that different queries do not retrieve the same information in different forms. And each query is directly relevant to the fact-checking.

    Your goal is to create an efficient, **concise**, and **effective** verification framework that maximizes accuracy while minimizing unnecessary complexity.

    ### **Output Format (Strict JSON Standard Required)**  
    Your response must follow this exact JSON structure:

    ```json
    {
    "reasoning_steps": [
        {
        "step": "[Step name]",
        "method": "[Verification method used]",
        "details": "[Detailed explanation of this reasoning step]"
        },
        {
        "step": "[Step name]",
        "method": "[Verification method used]",
        "details": "[Detailed explanation of this reasoning step]"
        },
        ...
    ],
    "validation_list": [
        {
        "sentence": "[Sentence 1] Keep in mind that this will be issued directly to a search engine, so the sentence should also contain all essential information explicitly, without relying on pronouns such as 'her' or 'his'. If there is no original sentence to validate, you can make the validation_list empty.",
        "explanation": "[Why this needs to be validated]"
        },
        {
        "sentence": "[Sentence 2] Keep in mind that this will be issued directly to a search engine, so the sentence should also contain all essential information explicitly, without relying on pronouns such as 'her' or 'his'. If there is no original sentence to validate, you can make the validation_list empty."
        "explanation": "[Why this needs to be validated]"
        }
    ],
    "search_list": [
        {
        "information_needed": "[What specific information needs to be found]",
        },
        {
        "information_needed": "[What specific information needs to be found]",
        }
    ]
    } 
    ```
    ## Note: 
    - The validation_list and search_list must not contain overlapping items.
    - Only include information that is truly worth searching for. If something can be easily inferred through common knowledge, do not list it. Make the number of information as short as possible, but not omit key information.
    - Each sentence in the validate list should also contain all essential information explicitly, without relying on pronouns such as 'her' or 'his'.
        """

    post_information = """
    Below is the post text, image and claim. 
    Post content: {}
    Claim: {}
    Post image: 
    """
    prompt = prompt_role + background + prompt_task  + post_information.format(post['post_text'], post['claim'] )  

    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]
    

    # 添加证据反馈
    if content_reasoning:
        content.append( {
                "type": "text",
                "text": "This is the previous reasoning result under your previous plan" + str(content_reasoning)  # 文本上下文
            })
    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content, image_path=None)
    return results, total_tokens


# ---------------------------------------------------------------------
# [Paper] Agent-TR — Text Evidence Retrieval and Validation Agent,
# sub-step 1/4: Query Generation (Sec. III-C.2).
# Turns the search list (L_s) into concrete search-engine queries (Q).
# 论文对应：文本证据检索验证智能体 Agent-TR 的子步骤(1/4)：查询生成
#（方法节 III-C.2）。把搜索意图清单(L_s)转化成可提交给搜索引擎的查询词(Q)。
#
# NOTE: the prompt below instructs the model to "avoid previously used
# queries" / "review the provided search history", but this function does
# not receive or pass any query-history context — that instruction is
# currently not backed by any actual history data.
# 注意：下面的 prompt 要求模型"避免重复之前用过的查询"/"参考已提供的搜索
# 历史"，但本函数并未接收或传入任何历史查询数据——这条指令目前没有实际
# 生效的数据支撑。
# ---------------------------------------------------------------------
def call_Query_Site_generator(search_list, post):
    print("*Step* Query_Site_generator")
    prompt_role = "You are an intelligent fact-checking assistant for a fact-checking framework."

    prompt_task = ("""Please analyze the given information need and generate a set of **SEO-friendly search queries** that rank well on Google. 
    Ensure that the queries are **high-intent, long-tail keywords** and can be directly used in search engines.
    Additionally, suggest **reliable websites** that are likely to provide relevant information.

    **Guidelines for Query Generation:**
    1. **Ensure queries contain high-intent and long-tail keywords** for precise results.
    2. **Avoid previously used queries** by reviewing the provided search history.
    3. **For complex searches, apply the divide-and-conquer strategy** to break down the request into **smaller, more searchable sub-information** and generate focused search queries.
    4. **Tailor search queries to fit the target websites’ language style** and expected content format.
    5. **Remove the similar query that have appeared in the previous query list.**
    **Search Optimization Techniques:**
    1. **Expected Keywords from Results:**  
       - Example: "moon is made of *", "lunar composition research paper", "NASA lunar surface chemical analysis"

    2. **Semantic Expansion (Synonym Expansion):**  
       - Example: "lunar composition OR moon material structure", "scientific study OR peer-reviewed research"

    3. **Keyword Variations:**  
       - Example: "lunar rock sample" vs. "lunar rock samples", "moon geology" vs. "lunar geology"

    4. **Question-Based Queries:**  
       - Example: "What is the chemical composition of the Moon?", "How was the Moon's geology analyzed?"

    5. **Long-Tail Keywords:**  
       - Example: "detailed analysis of Moon's surface chemistry", "recent lunar soil composition research findings"

    6. **Time & Numeric Keywords:**  
       - Example: "lunar mineral composition study 2023", "latest research on Moon's geology (last 5 years)"

    7. **Boolean Operator Optimization:**  
       - Example: "(lunar rock composition OR moon soil minerals) AND (recent study OR scientific paper)"

    8. **Contextual Keyword Optimization:**  
       - Example: "NASA research on Moon geology published in 2023", "Lunar soil chemical analysis using Apollo mission data"

    ---
    
    ### **Output Format (Strict JSON Standard Required)**
    Your response must follow this exact JSON structure:

    ```json
    [
        {
            "queries": [
                "Best possible search query 1",
                "Best possible search query 2",
                ...
            ]
        }
    ]
    ```

    **Important Notes:**
    - For each piece of information, generate no more than one search query. Include only information that is genuinely worth searching. If it can be reasonably inferred from common knowledge, omit it. Keep the list as short as possible without leaving out essential information.
    - Each query should consist only information relevant keywords, do not use irrelevant phrases like 'information about', 'details of', 'analysis about' etc. Avoid vague terms or pronouns (e.g., he, her, it).
    - The query will be sent directly to a search engine, so clarity, specificity and conciseness are critical. 
""")
    
    post_information = """
    Below is the claim, the post text and image. 
    Claim: {}
    Post content: {}
    Post image: 
    """
    prompt = prompt_role + background + prompt_task  + post_information.format(post['post_text'], post['claim'] )  

    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]
    
    for search_item in search_list:
        if search_item:
            content.append( {
                    "type": "text",
                    "text": """{ "Information to search":""" + str(search_item) + """}"""
            })
    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content, image_path=None)
    return results, total_tokens


# ---------------------------------------------------------------------
# [Paper] Agent-R — Reasoning Agent, batch-level intermediate reasoning
# (Sec. III-C.4; Workflow Step 3: Evidence Selection).
# Screens the current batch of retrieved evidence for relevance before it
# is carried into the final evidence-grounded verification step.
# 论文对应：推理智能体 Agent-R 的分批中间推理（方法节 III-C.4；工作流第3步：
# 证据筛选）。对当前批次检索到的证据做相关性筛选，供最终的"证据支撑核查"
# 步骤使用。
# ---------------------------------------------------------------------
def call_Middle_Reasoner(image_analysis_result, context, post):
    print("*Step* Middle_Reasoner")
    prompt_task= """You are an AI expert in fact-checking and claim verification. Your task is to analyze a given claim and determine its authenticity based on a structured reasoning process. You must strictly follow the provided reasoning logic and return the results in JSON format.
    ---

    ### **Input:**
    - **Claim**: The claim that needs verification.
    - **Post Context**: The content of the post to which the claim is related, including text, images, or external links.
    - **Reasoning Logic**: A structured validation plan that defines the verification steps.
    - **Searched Data**: Relevant sources, if available, such as fact-checking reports, official documents, or primary sources.
    - **Image Evidence Analysis Result**: The results of image-based analysis, including tampering and miscaptioning assessments.
    ---

    ## **Task Requirements**

    ### 1. Understand and interpret the claim before fact-checking.
    - Carefully read and extract the **core meaning** of the claim.
    - Paraphrase the key claim in your own words to demonstrate understanding.
    - This interpretation will help ensure your fact-check is aimed at the **claim itself**, not the surrounding post or meta-commentary.
    - the output of this step should be named as "my_understanding_of_claim" in your final answer.


    ### 2. Strictly Follow the Reasoning Logic
    - Adhere to the given logical validation plan.
    - Ensure all steps are executed in the given sequence.

    ### 3. Go Through Each Piece of Searched Information and relevant image analysis result One by One for Every Reasoning Step and decide the useful ones for this step, or for direct fact-checking the input claim.
    - Principle: 
        - Every step's analysis **must be backed by clear evidence** by referring evidence id and title. 
        - If **no direct evidence** is required for this step, explicitly state `"Evidence not required"` and explain why.
        - If relevant evidence not found, you can provide reliable evidence based on your knowledge (if you can), and their source and link.
        - If no evidence is relevant in this step of reasoning, state `"Relevent evidence not found"`. Note: You can only use `"Relevent evidence not found"` if each piece of evidence is useless for your analysis, otherwise, list the evidence you used for your analysis.

    ### 6. Provide a **final_sufficiency_confidence (1-5)** to approve or refute this claim:
            5 = Excellent: We already have **strong and consistent evidence**. New evidence is unlikely to alter the conclusion.
            4 = Good: The reasoning logic is well-supported by current evidence. While additional evidence might refine the assessment, it is unlikely to cause a major shift.
            3 = Moderate: There is some supporting evidence, but it is not strong enough to definitively establish the authenticity type. More evidence is needed for confirmation.
            2 = Weak: The available evidence is limited and leaves significant gaps in verification, making it difficult to establish the authenticity type.
            1 = Very Weak: The evidence is minimal, unreliable, or highly questionable, making it nearly impossible to determine the authenticity type.

    **Note**: 
    1. Go through each input evidence for each step and decide whether this one is useful for this step.
    2. Pay close attention to the reliability of each evidence source at every step. Carefully evaluate the credibility of conflicting sources when discrepancies arise, and weigh them appropriately in your final judgment.
    3. Do not use the text or image content from the input post as evidence to support the claim. Since the post is the subject of fact-checking, its trustworthiness is unverified and therefore cannot be considered a valid source.
    4. Be cautious and modest in giving a confidence score. The fact that no evidence has been found so far does not mean the claim is entirely false. 
    ---

    ### **🔹 Updated JSON Output Format**
    ```json
    {
        "my_understanding_of_claim": "Your paraphrased version of the key fact-checking or misleading information in the claim here.",
        "validation_result": {
            "reasoning_steps": [
                {
                    "step_name": "<STEP_NAME_FROM_INPUT>",
                    "description": "<Description of this step>",
                    "analysis_result": "<Reliable analysis result>",
                    "relevant_evidence_summary": "<Summary of relevant evidence OR 'Evidence not required' OR 'Relevent evidence not found'>",
                    "relevant_text_evidence_list"(If 'Relevent evidence not found' or 'Evidence not required' for this step, set the value here empty as []): [
                        {
                            "id": "<id of source> (consistent with the id in the information list)",
                            "title": "<Title of source>",
                            "is_used": "<true>",
                            "source_reputation": "good/bad/normal"
                        }
                    ], 
                    "relevant_image_evidence_list"(If 'Relevent evidence not found' or 'Evidence not required', set it empty as []):[ 
                        "evidence_0 (consistent with the item in the image analysis result list)":{
                        "image_tampering_analysis_result": "<0-100%>",
                        "image_miscaption_analysis_result": "<0-100%>",
                        },
                        "evidence_1":{
                        "image_tampering_analysis_result": "<0-100%>",
                        "image_miscaption_analysis_result": "<0-100%>",
                        },
                    ],
                    "evidence_based_on_my_knowldge":[
                        "title": "<Title of source>",
                        "content": "<Key information of the evidence>",
                        "link": "<Link of the evidence>",
                        "source": "<Platform and author of the evidence>",
                        "is_used": "<true>",
                    ]
            ],
            "direct_fact_check_evidence":{ 
                "analysis_result": "<Reliable analysis result>",
                "relevant_evidence_summary": "<Summary of relevant evidence OR 'Evidence not required' OR 'Relevent evidence not found'>",
                "relevant_text_evidence_list":[
                    {
                        "id": "<id of source> (consistent with the id in the information list or image analysis result list)",
                        "title": "<Title of source>",
                        "is_used": "<true>",
                        "source_reputation": "good/bad/normal"
                    }
                ]
            }
            "final_sufficiency_confidence": "<Final confidence score (1-5)>"
        }
    }

    """
    post_information = """
        Below is the post text, image and claim. 
        Post content: {}
        Claim: {}
        Post image: 
        """

    prompt =  background + prompt_task + post_information.format(post['post_text'], post['claim'] ) 


    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]

    # 添加上下文
    if context:
        content.append( {
                "type": "text",
                "text": "The following is the provided reasoning logic (plan) and the information list searched by an information search agent group:"+ (str(context) if context else '') + ". The following is the image analysis result provided by an image analysis agent group. " +str(image_analysis_result)  # 文本上下文
            })
        prompt = prompt + "The following is the provided reasoning logic (plan) and the information list searched by an information search agent group:"+ (str(context) if context else '') + ". The following is the image analysis result provided by an image analysis agent group. " +str(image_analysis_result)

    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content, image_path=None)
    return results, total_tokens


# ---------------------------------------------------------------------
# [Paper] Agent-R — Reasoning Agent, final evidence-grounded reasoning
# (Sec. III-C.4; Workflow Step 4: Evidence-Grounded Verification).
# Rephrases the claim first ("my_understanding_of_claim") to stay focused,
# then reasons step-by-step over all accumulated evidence and outputs a
# sufficiency confidence score (1-5).
# 论文对应：推理智能体 Agent-R 的最终证据支撑推理（方法节 III-C.4；工作流
# 第4步：证据支撑核查）。先复述claim("my_understanding_of_claim")避免推理
# 跑偏，再逐步推理所有累积证据，输出信息充分度置信分数(1-5)。
# ---------------------------------------------------------------------
def call_Middle_Reasoner_with_Source_Judgment(image_analysis_result, context, post):
    print("*Step* Middle_Reasoner")
    prompt_task= """You are an AI expert in fact-checking and claim verification. Your task is to analyze a given claim and determine its authenticity based on a structured reasoning process. You must strictly follow the provided reasoning logic and return the results in JSON format.
    ---

    ### **Input:**
    - **Claim**: The claim that needs verification.
    - **Post Context**: The content of the post to which the claim is related, including text, images, or external links.
    - **Reasoning Logic**: A structured validation plan that defines the verification steps.
    - **Searched Data**: Relevant sources, if available, such as fact-checking reports, official documents, or primary sources.
    - **Image Evidence Analysis Result**: The results of image-based analysis, including tampering and miscaptioning assessments.

    ---

    ## **Task Requirements**

    ### 1. Understand and interpret the claim before fact-checking.
    - Carefully read and extract the **core meaning** of the claim.
    - Paraphrase the key claim in your own words to demonstrate understanding.
    - This interpretation will help ensure your fact-check is aimed at the **claim itself**, not the surrounding post or meta-commentary.
    - the output of this step should be named as "my_understanding_of_claim" in your final answer.


    ### 2. Strictly Follow the Reasoning Logic
    - Adhere to the given logical validation plan.
    - Ensure all steps are executed in the given sequence.

    ### 3. Go Through Each Piece of Searched Information and relevant image analysis result One by One for Every Reasoning Step and decide the useful ones for this step, or for direct fact-checking the input claim.
    - Principle: 
        - Every step's analysis **must be backed by clear evidence** by referring evidence id and title. 
        - If **no direct evidence** is required for this step, explicitly state `"Evidence not required"` and explain why.
        - If relevant evidence not found, you can provide reliable evidence based on your knowledge (if you can), and their source and link.
        - If no evidence is relevant in this step of reasoning, state `"Relevent evidence not found"`. Note: You can only use `"Relevent evidence not found"` if each piece of evidence is useless for your analysis, otherwise, list the evidence you used for your analysis.

    ### 6. Provide a **final_sufficiency_confidence (1-5)** to approve or refute this claim:
            5 = Excellent: We already have **strong and consistent evidence**. New evidence is unlikely to alter the conclusion.
            4 = Good: The reasoning logic is well-supported by current evidence. While additional evidence might refine the assessment, it is unlikely to cause a major shift.
            3 = Moderate: There is some supporting evidence, but it is not strong enough to definitively establish the authenticity type. More evidence is needed for confirmation.
            2 = Weak: The available evidence is limited and leaves significant gaps in verification, making it difficult to establish the authenticity type.
            1 = Very Weak: The evidence is minimal, unreliable, or highly questionable, making it nearly impossible to determine the authenticity type.

    **Note**: 
    1. Go through each input evidence for each step and decide whether this one is useful for this step.
    2. Pay close attention to the reliability of each evidence source at every step. Carefully evaluate the credibility of conflicting sources when discrepancies arise, and weigh them appropriately in your final judgment. 
    3. Do not use the text or image content from the input post as evidence to support the claim. Since the post is the subject of fact-checking, its trustworthiness is unverified and therefore cannot be considered a valid source.
    4. Be cautious and modest in giving a confidence score. The fact that no evidence has been found so far does not mean the claim is entirely false. 
    ---

    ### **🔹 Updated JSON Output Format**
    ```json
    {
        "my_understanding_of_claim": "Your paraphrased version of the key fact-checking or misleading information in the claim here.",
        "validation_result": {
            "reasoning_steps": [
                {
                    "step_name": "<STEP_NAME_FROM_INPUT>",
                    "description": "<Description of this step>",
                    "analysis_result": "<Comprehensive analysis result of the evidence list>",
                    "relevant_evidence_summary": "<Summary of relevant evidence OR 'Evidence not required' OR 'Relevent evidence not found'>",
                    "relevant_text_evidence_list"(If 'Relevent evidence not found' or 'Evidence not required' for this step, set the value here empty as []): [
                        {
                            "id": "<id of source> (consistent with the id in the information list)",
                            "title": "<Title of source>",
                            "is_used": "<true>"
                            "source_reputation": "reliable/unreliable/satire/unsure"
                        }
                    ], 
                    "relevant_image_evidence_list"(If 'Relevent evidence not found' or 'Evidence not required', set it empty as []):[ 
                        "evidence_0 (consistent with the item in the image analysis result list)":{
                        "image_tampering_analysis_result": "<0-100%>",
                        "image_miscaption_analysis_result": "<0-100%>",
                        "is_used": "<true>"
                        "source_reputation": "reliable/unreliable/satire/unsure"
                        },
                        "evidence_1":{
                        "image_tampering_analysis_result": "<0-100%>",
                        "image_miscaption_analysis_result": "<0-100%>",
                        "is_used": "<true>"
                        "source_reputation": "reliable/unreliable/satire/unsure"
                        },
                    ],
                    "evidence_based_on_my_knowldge":[
                        "title": "<Title of source>",
                        "content": "<Key information of the evidence>",
                        "link": "<Link of the evidence>",
                        "source": "<Platform and author of the evidence>",
                        "is_used": "<true>",
                        "source_reputation": "reliable/unreliable/satire/unsure"
                    ]
            ],
            "direct_fact_check_evidence":{ 
                "analysis_result": "<Reliable analysis result>",
                "relevant_evidence_summary": "<Summary of relevant evidence OR 'Evidence not required' OR 'Relevent evidence not found'>",
                "relevant_text_evidence_list":[
                    {
                        "id": "<id of source> (consistent with the id in the information list or image analysis result list)",
                        "title": "<Title of source>",
                        "is_used": "<true>",
                        "source_reputation": "reliable/unreliable/satire/unsure"
                    }
                ]
            }
            "final_sufficiency_confidence": "<Final confidence score (1-5)>"
        }
    }

    """
    post_information = """
        Below is the post text, image and claim. 
        Post content: {}
        Claim: {}
        Post image: 
        """

    prompt =  background + prompt_task + post_information.format(post['post_text'], post['claim'] ) 


    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]

    # 添加上下文
    if context:
        content.append( {
                "type": "text",
                "text": "The following is the provided reasoning logic (plan) and the information list searched by an information search agent group:"+ (str(context) if context else '') + ". The following is the image analysis result provided by an image analysis agent group. " +str(image_analysis_result)  # 文本上下文
            })
        prompt = prompt + "The following is the provided reasoning logic (plan) and the information list searched by an information search agent group:"+ (str(context) if context else '') + ". The following is the image analysis result provided by an image analysis agent group. " +str(image_analysis_result)

    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content)
    return results, total_tokens

# ---------------------------------------------------------------------
# [Paper] Agent-EG — Explanation Generation Agent
# (Sec. III-C.5; Workflow Step 5: Explanation Generation).
# Produces the final structured output: veracity label (2-class and
# 3-class), reasoning summary, evidence-ID-cited key points, and a
# confidence score.
# 论文对应：解释生成智能体 Agent-EG（方法节 III-C.5；工作流第5步：解释
# 生成）。产出最终结构化结果：真伪标签（2分类和3分类）、推理摘要、引用
# 证据ID的关键推理点、置信度分数。
# ---------------------------------------------------------------------
def call_Explainer_3_class_aligned(img_evi_analysis, context, content_reasoning, post):
    print("*Step* Explainer")
    prompt_role = "You are a Fact-Checking Result Explainer for a fact-checking framework."
    prompt_task="""
Based on the provided context (which could include the target claim that expected to be fact-checked, its post context, the searched evidence, and the previous reasoning), perform a **final analysis** and provide the most reliable **authenticity and motivation assessment** of the claim.

### **Your task includes:**
1. **Understand and interpret the claim before fact-checking.
   - Carefully read and extract the **core meaning** of the claim.
   - Paraphrase the claim in your own words to demonstrate understanding.
   - This interpretation will help ensure your fact-check is aimed at the **claim itself**, not the surrounding post or meta-commentary.
   - the output of this step should be named as "my_understanding_of_claim" in your final answer.

2. **Evaluate the claim’s authenticity** based on the provided evidence and reasoning process.
   - **IMPORTANT**: Your analysis should focus on validating the **claim itself**, not the post.  
     - Example: If the claim states a post is "false," and the post is actually false, then your authenticity label should be **"TRUE"**, because the claim itself is correct.
    - IMPORTANT: At every step, carefully assess the reliability of each evidence source. Sources with a strong reputation are more likely to provide accurate and trustworthy information, while sources with a bad reputation are much more likely to be false or misleading.
   - Justify the claim's authencity by selecting its coarse type and fine-grained type.
   - Explain the key **reasoning points** behind your judgment.
   - Indicate whether more external information is still needed for further improvement.

3.  Definition: coarse type include {TRUE, FALSE} and fine-grained type type include {UNPROVEN, TRUE, FALSE}  referring to the following definietions of each type.
   - "true": There is strong evidence from reliable sources that confirms the claim.
   - "false": There is strong evidence from reliable sources that clearly refutes or contradicts the claim.
   - "unproven": There is no sufficient evidence to confirm or refute the claim. Lack of evidence does NOT mean false.

4. **Decision process**:
   - First, check whether any credible evidence directly refutes the claim.
       - If such evidence exists and is strong/reliable → label as "false".
   - If no refuting evidence exists, but strong confirming evidence exists → label as "true".
   - If neither confirming nor refuting evidence is found, OR the evidence is weak / uncertain / inconclusive → label as "unproven".

5. **Assess the sufficiency of the current evidence and reasoning**:
   - Assign a **Confidence Score (1-5)** to indicate how certain your judgment is.
     - **5**: Highly confident, reliable and multi-sourced supporting evidence.
     - **3**: Moderate confidence, reliable evidence but the source richness is unsure.
     - **1**: Low confidence, requires more evidence.

6. **List all evidence used in your reasoning**:
   - Cite **the evidence id** for your reasoning key points. The evidence id should be consistent with the evidence in the provided evidence list.
   - Cite **image_evidence_analysis id*  for their supported reasoning key points. The image_evidence_analysis should be consistent with the #input# image evidence analysis. 

### **Output Format**
Return your results in the following structured JSON format:

```json
{
    "my_understanding_of_claim": "Your paraphrased version of the claim here.",
    "validation_result": {
        "2-class_authenticity_label": "choose from <FALSE, TRUE>",
        "3-class_authenticity_label": "choose from <FALSE, TRUE, UNPROVEN>",
        "reasoning_logic": "Explain your final reasoning process based on all available context.",
        "key_points": [
        "1. XXX (cite the evidence IDs such as '1', '2', '3' that support each point if available; cite the image_evidence_analysis's id such as 'evidence_0', 'evidence_1' if it supports this point).",
        "2. XXX",
        "3. XXX"
        ]
    },
  "confidence_level": "<1-5>"
}

**Note**: Your analysis should focus on assessing the authenticity of the core claim itself, not the post, or unimportant information. For example, if the claim asserts that a post is “false,” and the post is indeed false, then the authenticity label should be "TRUE"—because the claim is correct.


Now, begin your reasoning for the next claim."""
    
    post_information = """
    # Input:
        Below is the post text, image and claim. 
        Post content: {}
        Claim: {}
        Post image: 
        """

    prompt = prompt_role+background+ prompt_task + post_information.format(post['post_text'], post['claim'] ) 


    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]

    # 添加上下文
    if context:
        content.append( {
                "type": "text",
                "text": "The following is the current evidence we have:"+ str(context) # 文本上下文 
            })
        prompt = prompt + "The following is the current evidence we have:"+ str(context)
    if img_evi_analysis:
        content.append( {
                "type": "text",
                "text": "The following is the current image analysis result:"+ "{'KEY': ImageAnalysis, 'content': " +str(img_evi_analysis)
            })
        prompt = prompt + "The following is the current image analysis result:"+ "{'KEY': ImageAnalysis, 'content': " +str(img_evi_analysis)
    if content_reasoning:
        content.append( {
                "type": "text",
                "text": "Here is the reasoning process based on current evidence and image analysis result"+ str(content_reasoning)
            })
        prompt = prompt + "Here is the reasoning process based on current evidence and image analysis result"+ str(content_reasoning)


    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content)
    return results, total_tokens


#####################图片分析系列函数#####################

# ---------------------------------------------------------------------
# [Paper] Agent-IR — Image Retrieval and Analysis Agent, relevancy/
# text-only pre-filtering (Sec. III-C.3).
# Checks whether the post image is text-only or relevant to the claim.
# 论文对应：图像检索分析智能体 Agent-IR 的相关性/纯文字图片预筛选
#（方法节 III-C.3）。判断帖子图片是否为纯文字图片，或是否与claim相关。
#
# NOTE: NOT called from main_workflow.py (not imported there) — currently
# unused / not wired into the pipeline.
# 注意：main_workflow.py 里没有 import 或调用这个函数——当前未接入主流程。
# ---------------------------------------------------------------------
def call_ImageRelevancyJudger(post):
    print("*step* Image Claim Relevancy Judger")
    prompt_role = "You are a Text-Image Relevancy Judger for a fact-checking framework."

    prompt_task = """
    Your task is to decide whether the image is relevant to the claim or the post text by following these steps:

    0. **Extract the text from the image (If available).
    1. **Determine if the image is text-only:**
        - Check if the image's main subject consists solely of text (e.g., a screenshot of text-only Twitter content). 
        - Be strict when making the "text-only image" determination: Recap the image and decide whether there is any meaningful visual (non-text) information present in the image. If the image contains a mix of text and visuals (e.g., half text, half image) or has text overlaid on a photo, it does NOT qualify as a "text-only image." 
        - If yes, "result": "text-only image". 
        - If no, proceed to Step 2.

    2. **Assess the image's relevancy to the claim/post text:**
        - Determine if the image event is relevant to the claim or post text (note: don't be strict).
            - If no, "result": "claim irrelevant image".
            - If yes, proceed to Step 3.

    3. **Evaluate the image's influence on claim authenticity:**
    - Determine whether the image's authenticity or truthfulness is relevant to the authenticity of the claim or post text.
    - Specifically, evaluate whether the image is intended to serve as visual evidence in the context of the original post. 
        - Focus solely on the image's intended purpose in supporting or refuting the claim, as implied by the post.
        - Do NOT evaluate the actual quality, accuracy, or credibility of the image itself—consider only its intended role in the post.
    - Provide your judgment and explanation in the following format:
        ```json
        {
            "result": "claim relevant image",
            "claim_authenticity_relevant": "<Yes or No>",
            "reasoning": "<Provide detailed reasoning>"
        }
        ```
    """

    answer_format = """
        The final JSON response format should be:
        {
            "result": "<text-only image | claim irrelevant image | claim relevant image>",
            "extracted_text": "<text extracted from the image (if applicable)>",
            "claim_authenticity_relevant": "<Yes or No (if applicable)>",
            "reasoning": "<Detailed reasoning (if applicable)>"
        }
    """

    prompt = prompt_role + "\n\n" + prompt_task + "\n\n" + answer_format


    post_information = """
        Below is the post text, image and claim. 
        Post content: {}
        Claim: {}
        Post image: 
        """

    prompt = prompt_role+ prompt_task + answer_format + post_information.format(post['post_text'], post['claim'] ) 

    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]

    #添加post图像到消息content
    image = post['post_image']  # post_image is a single path string, not a list
    if image:
        try:
            # if "./" in image:
            #     image = image.replace(".", '/home/danni/projects/LMM-misinformation/codes/groundtruth_generation', 1)
            # 将每个图片添加为base64数据
            code_image = image_to_code(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{code_image}"
                    }
                }
            )
        except:
            print(f"the image in the path {image} is not correctly open or encoded")
            return None
    
    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content, image_path=None)
    return results, total_tokens        
  
 
# ---------------------------------------------------------------------
# [Paper] Agent-IR — Image Retrieval and Analysis Agent: image
# categorization + visual tampering detection
# (Sec. III-C.3; Workflow Step 2: Evidence Acquisition).
# Classifies the (post image, retrieved image) pair into one of three
# relationship types and estimates a tampering probability.
# 论文对应：图像检索分析智能体 Agent-IR 的图像分类与篡改检测
#（方法节 III-C.3；工作流第2步：证据获取）。把（帖子图片，检索到的证据
# 图片）分类为三种关系类型之一，并估计篡改概率。
# ---------------------------------------------------------------------
def call_image_similarity_type_and_manipulation_judger(page_result, post):
    print("*step* image_similarity_type_and_manipulation_judger")
    evid_image = page_result['img_address']
    post_text = post['post_text']
    claim = post['claim']
    # Prompt Template
    prompt = """
    You are an advanced similar images comparison assistant for a fact-checking company. You will be provided with an post image that requires fact-checking and an online searched image which could potentially serve as an evidence. Your task is to carefully identify the similarity and difference between two images, and the possibility of the post image is problematic.
    Here are the steps that you can refer to.
    Step 1: Determine the relationship between the two images: a post image (provided in a social media post) and an evidence image (potentially related or unrelated). Based on your analysis, classify their relationship into one of the following categories:
    1. **Potentially From Same Source**: The post image and evidence image are nearly identical (picture of same contents, same actions and same configuration, potentially from the same original raw image).
    2. **Same Event, Different Content**: The post image and evidence image depict the same real-world event but differ in visual content (e.g., taken from different angles, at different times, or by different cameras).
    3. **No Close Relationship**: The post image and evidence image have no significant connection (e.g., depict different events, objects, or subjects entirely).

    (If the two images are "No Close Relationship" skip the next step 2.)
    Step 2: If the two images are not "No Close Relationship", Evaluate whether the post image has been tampered by both analyzing the image-itself and comparing with the evidence image. Detect any visual differences or alterations between the post image and the evidence image (e.g., added or removed elements, including misleadingly adding or removing texts or visual elements, cropping images and other).""

    ### Output:
    Your output should follow this structured JSON format:
    {
        "relationship": "<Potentially From Same Source | Same Event, Different Content | No Close Relationship>",
        "relationship_Reasoning": "<Explain your analysis in detail to justify the chosen category>",
        "tampering_probability": "A likelihood score for whether the post image has been tampered compared to the evidence image (0–100) (If the relationship is no close relationship, leave this item empty)",
        "tampering_reasoning": "Use less than 50 words to explain your analysis, referencing: Image self-analysis results and image comparison results. (If the relationship is no close relationship, leave this item empty)",
        "confidence": "A numeric score (0–100) reflecting the overall confidence in your analysis."
    }
    ### Additional Notes:
    - Compare the images in detail, especially pay attention to distinguishing features and claim-relevant features between them.
    - Be precise in your evaluation and provide clear reasoning for your scores.
    """


    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt + "This is the post text" + post_text + ". This is the main claim in the post " +claim + ". post image:" # 你原来的文本输入
        }
    ]

    #添加post图像到消息content
    image = post['post_image']  # post_image is a single path string, not a list
    if image:
        try:
            # if "./" in image:
            #     image = image.replace(".", '/home/danni/projects/LMM-misinformation/codes/groundtruth_generation', 1)
            # 将每个图片添加为base64数据
            code_image = image_to_code(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{code_image}"
                    }
                }
            )
        except:
            print(f"the image in the path {image} is not correctly open or encoded")
            return None
    
    content.append({
            "type": "text",
            "text": "This is the evidence image"  # 你原来的文本输入
        })

    #添加证据图像到content
    if evid_image:
        try:
            # if "./" in image:
            #     image = image.replace(".", '/home/danni/projects/LMM-misinformation/codes/groundtruth_generation', 1)
            # 将每个图片添加为base64数据
            code_image = image_to_code(evid_image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{code_image}"
                    }
                }
            )
        except:
            print(f"the image in the path {evid_image} is not correctly open or encoded")
            return None
    else:
        print("there is no evidence image")
        return ''

    
    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content, image_path=None)
    return results, total_tokens

# ---------------------------------------------------------------------
# [Paper] Agent-IR — Image Retrieval and Analysis Agent: miscaption
# detection (Sec. III-C.3; Workflow Step 2: Evidence Acquisition).
# Checks whether the claim/post text is consistent with the retrieved
# webpage text associated with a visually similar image.
# 论文对应：图像检索分析智能体 Agent-IR 的图文不符检测
#（方法节 III-C.3；工作流第2步：证据获取）。检查claim/帖子文本与检索到的
# 相似图片所在网页文本是否一致。
# ---------------------------------------------------------------------
def call_image_miscaption_detector(page_result, post, post_image_text):
    print("*step* image_miscaption_detector")
    # evid_image = page_result['img_address']
    title = page_result['title']
    body = page_result['body']
    post_text = post['post_text']
    claim = post['claim']

    prompt = """
    You are an advanced image-text miscaption analysis assistant. Your task is to evaluate whether an image has been **misused or miscaptioned** (e.g., repurposed in a misleading context) to support a **social media post's claim**.

    This task involves analyzing the relationship between a post image and its associated claim. You will be provided with the post's **claim**, **image**, **original text**, and **OCR-extracted text (if available)**, along with an **evidence image** and its contextual information.

    ---
    ### Key Steps:

    1. **Understand and interpret the claim before fact-checking**
    - Carefully read and extract the **core meaning** of the claim (avoid focusing on emotional tone or rhetorical flourishes).
    - Paraphrase the claim in your own words to confirm understanding.
    - This ensures the fact-check targets the actual assertion, not surrounding commentary.

    → Output this step as: `"my_understanding_of_claim"`

    ---

    2. **Understand the evidence**
    - Rephrase the evidence content in your own words.
    - Identify the **context and purpose** of the evidence image.
    - Determine if the evidence image was taken from the same event, time, place, and subject described in the claim.

    ---

    3. **Extract temporal and contextual information (if available)**
    - Compare time-related details (e.g., timestamps, dates, event names) in the **post** vs the **evidence**.
    - Note whether the temporal gap (if any) creates a **misleading impression** or changes the meaning of the image.

    ---

    4. **Miscaption Analysis**
    #### Definition:
    > Miscaptioning occurs **only when** an image is used to directly support a **factually false claim**, or when the image **substantially misrepresents the time, place, people, or context** of an event.
    #### Do **not** consider the following as miscaptioning:
    - The post uses emotional, biased, or opinionated language  
    - The post omits certain background details  
    - The image comes from the same event but is used to support a **simplified or one-sided interpretation**

    #### Do consider as miscaptioning:
    - The image is reused from a **completely unrelated event**, place, or time  
    - The image misrepresents who is involved or **what is happening**
    ---

    ### Input Format:

    - **Post**:
    - **Claim**: <Key claim made in the post>
    - **Image**: <Image in the post>
    - **Text**: <Text associated with the post>

    - **Evidence**:
    - **Image**: <Image from the evidence>
    - **Text**: <Text associated with the evidence>

    ---

    ### Output Format:

    {
        "my_understanding_of_claim": "<Your paraphrased version of the claim>",
        "Miscaption Rate": "<A likelihood score from 0–100 indicating how likely the image misrepresents the claim>",
        "Reasoning": "<Use less than 50 words to explain whether the image was fairly used to support the claim. Focus on factual alignment. Missing nuance is not enough. Only assign a high score if the image directly misleads about the core facts.>"
    }

    ---

    ### Scoring Guide:

    | Score Range | Meaning |
    |-------------|---------|
    | **0–20**    | The image accurately supports the claim; no miscaptioning  
    | **30–50**   | The image aligns broadly but lacks key context or nuance  
    | **60–80**   | The image gives a **misleading impression** of the facts or event  
    | **90–100**  | The image is **false, unrelated, or contradicts** the claim in a major way  
    ---

    ### Final Note:
    Your final answer should focus on **fact-based consistency** between the post's image and its claim. Avoid over-penalizing emotional tone, oversimplification, or lack of full context **unless they cause the image to clearly mislead viewers about the core facts**.
    """

    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt + " This is claim of the post text:" + claim + "This is the original post text: " + post_text + ". This is the text extracted from the post image via OCR (if available):" + post_image_text+ ". This is the post image:" # 你原来的文本输入
        }
    ]

    #添加post图像到消息content
    image = post['post_image']  # post_image is a single path string, not a list
    if image:
        try:
            # if "./" in image:
            #     image = image.replace(".", '/home/danni/projects/LMM-misinformation/codes/groundtruth_generation', 1)
            # 将每个图片添加为base64数据
            code_image = image_to_code(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{code_image}"
                    }
                }
            )
        except:
            print(f"the image in the path {image} is not correctly open or encoded")
            return None
    
    content.append({
            "type": "text",
            "text": "This is the evidence web's title and content. Title: " + title + ". Content:" + body + ". This is the evidence image"  # 你原来的文本输入
        })

    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content, image_path=None)
    return results, total_tokens


# ---------------------------------------------------------------------
# [Paper] Earlier/alternate draft of Agent-R's reasoning step (compare
# with call_Middle_Reasoner_with_Source_Judgment above).
# NOT called from main_workflow.py — currently unused / not wired into
# the pipeline; kept for reference only.
# 论文对应：Agent-R 推理步骤的较早/备用草稿版本（对比上面的
# call_Middle_Reasoner_with_Source_Judgment）。main_workflow.py 里没有
# 调用这个函数——当前未接入主流程，仅作历史参考保留。
# ---------------------------------------------------------------------
def source_analyzer(image_analysis_result, context, post):
    print("*Step* Middle_Reasoner")
    prompt_task= """You are an AI expert in fact-checking and claim verification. Your task is to analyze a given claim and determine its authenticity based on a structured reasoning process. You must strictly follow the following steps and return the results in JSON format.
---

### **Input:**
- **Claim**: The claim that needs verification.
- **Post Context**: The content of the post to which the claim is related, including text, images, or external links.
- **Searched Data**: Relevant sources, if available, such as fact-checking reports, official documents, or primary sources.
- **Image Evidence Analysis Result**: The results of image-based analysis, including tampering and miscaptioning assessments.

---

## **Task Requirements**

### 1. Understand and interpret the claim before fact-checking.
   - Carefully read and extract the **core meaning** of the claim.
   - Paraphrase the key claim in your own words to demonstrate understanding.
   - This interpretation will help ensure your fact-check is aimed at the **claim itself**, not the surrounding post or meta-commentary.
   - the output of this step should be named as "my_understanding_of_claim" in your final answer.


### 2. Go Through Each Piece of Searched Information and relevant image analysis result One by One and decide the useful ones for fact-checking the input claim.
   - Principle: 
    - Analysis **must be backed by clear evidence** by referring evidence id and title. 
    - If **no direct evidence** is required, explicitly state `"Evidence not required"` and explain why.
    - If relevant evidence not found, you can provide reliable evidence based on your knowledge (if you can), and their source and link.
    - If no evidence is relevant for reasoning, state `"Relevent evidence not found"`. Note: You can only use `"Relevent evidence not found"` if each piece of evidence is useless for your analysis, otherwise, list the evidence you used for your analysis.

### 6. Provide a **final_sufficiency_confidence (1-5)** to approve or refute this claim:
        5 = Excellent: We already have **strong and consistent evidence**. New evidence is unlikely to alter the conclusion.
        4 = Good: The reasoning logic is well-supported by current evidence. While additional evidence might refine the assessment, it is unlikely to cause a major shift.
        3 = Moderate: There is some supporting evidence, but it is not strong enough to definitively establish the authenticity type. More evidence is needed for confirmation.
        2 = Weak: The available evidence is limited and leaves significant gaps in verification, making it difficult to establish the authenticity type.
        1 = Very Weak: The evidence is minimal, unreliable, or highly questionable, making it nearly impossible to determine the authenticity type.

**Note**: 
1. Go through each input evidence for each step and decide whether this one is useful for this step.
2. Pay close attention to the reliability of each evidence source at every step. Carefully evaluate the credibility of conflicting sources when discrepancies arise, and weigh them appropriately in your final judgment.
3. Do not use the text or image content from the input post as evidence to support the claim. Since the post is the subject of fact-checking, its trustworthiness is unverified and therefore cannot be considered a valid source.
4. Be cautious and modest in giving a confidence score. The fact that no evidence has been found so far does not mean the claim is entirely false. 
---

### **🔹 Updated JSON Output Format**
```json
{
    "my_understanding_of_claim": "Your paraphrased version of the key fact-checking or misleading information in the claim here.",
    "validation_result": {
        "fact_check_evidence":{ 
            "analysis_result": "<Reliable analysis result>",
            "relevant_evidence_summary": "<Summary of relevant evidence OR 'Evidence not required' OR 'Relevent evidence not found'>",
            "relevant_text_evidence_list":[
                {
                    "id": "<id of source> (consistent with the id in the information list or image analysis result list)",
                    "title": "<Title of source>",
                    "is_used": "<true>",
                    "source_reputation": "good/bad/normal"
                }
            ]
        }
        "final_sufficiency_confidence": "<Final confidence score (1-5)>"
    }
}

"""
    post_information = """
        Below is the post text, image and claim. 
        Post content: {}
        Claim: {}
        Post image: 
        """

    prompt =  background + prompt_task + post_information.format(post['post_text'], post['claim'] ) 


    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]
    
   # 添加上下文
    if context:
        content.append( {
                "type": "text",
                "text": "The following is the provided reasoning logic (plan) and the information list searched by an information search agent group:"+ (str(context) if context else '') + ". The following is the image analysis result provided by an image analysis agent group. " + str(image_analysis_result)  # 文本上下文
            })
        prompt = prompt + "The following is the provided reasoning logic (plan) and the information list searched by an information search agent group:"+ (str(context) if context else '') + ". The following is the image analysis result provided by an image analysis agent group. " +str(image_analysis_result)

    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content)
    return results, total_tokens


# ---------------------------------------------------------------------
# [Paper] Earlier/simpler draft of Agent-TR's source-reliability sub-step
# (URL only, no page content). Superseded by
# call_evidence_source_with_link_judger_detailed below.
# NOT called from main_workflow.py — currently unused / not wired into
# the pipeline; kept for reference only.
# 论文对应：Agent-TR 信源可信度评估子步骤的较早/简化草稿版本（只看URL，
# 不抓取网页内容），已被下面的 call_evidence_source_with_link_judger_detailed
# 取代。main_workflow.py 里没有调用这个函数——当前未接入主流程，仅作历史
# 参考保留。
# ---------------------------------------------------------------------
def call_evidence_source_judger_detailed(source):
    print("*step* evidence_source_judger")

    # Embedding Input into Prompt
    prompt_template = """
You are an expert in digital literacy and online misinformation analysis.
Your task is to assess the reliability and intent of a given website URL.
Evaluate whether the website is a legitimate news or information source, a known disinformation or propaganda site, a parody/satirical outlet, or an untrustworthy source.
Steps:
1. Identify the domain (e.g., cnn.com, theonion.com, infowars.com).
2. Check for known flags: Listed on misinformation databases, Frequently debunked by fact-checking organizations, Known satire or parody site
3. Provide a concise final judgment and explain your reasoning clearly in 2–4 sentences.
    """

    prompt = prompt_template + """ ### Output Format:
    {
        "source_identification": "Example.com",
        "type": "<unreliable|satire|reliable|unsure>"
        "reasoning":"your judgment and reasoning"
    }
    Perform your analysis and provide the output in JSON format based on the given input.
    Here is the web to assess:
    """ + source

    # Output the final prompt
    # print(prompt)

    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]

    # 构建多个agents的对话信息
    results, _ =  llm_process(content, image_path=None)
    
    return results
  


# ---------------------------------------------------------------------
# [Paper] Agent-TR — Text Evidence Retrieval and Validation Agent,
# sub-step 4/4: Source Reliability Estimation (Sec. III-C.2).
# Classifies each evidence source into one of four categories: reliable,
# unreliable, satire, or unsure.
# 论文对应：文本证据检索验证智能体 Agent-TR 的子步骤(4/4)：信源可信度
# 评估（方法节 III-C.2）。把每条证据来源分类为四类之一：reliable(可信)/
# unreliable(不可信)/satire(讽刺)/unsure(不确定)。
#
# NOTE: this same function is also reused inside main_workflow.py's
# Agent-IR (image evidence) branch to judge the reliability of pages
# found via reverse image search — i.e. it serves both Agent-TR (text
# evidence) and Agent-IR (image-page evidence) roles.
# 注意：main_workflow.py 里 Agent-IR（图像证据）分支也复用了这同一个
# 函数，用来判断反向图片搜索命中的网页的可信度——也就是说这个函数同时
# 承担了 Agent-TR（文本证据）和 Agent-IR（图片页面证据）两个角色。
# ---------------------------------------------------------------------
def call_evidence_source_with_link_judger_detailed(source):
    print("*step* evidence_source_judger")

    # Embedding Input into Prompt
    prompt_template = """
You are an expert in digital literacy and online misinformation analysis.
Your task is to assess the reliability and intent of a given website URL.
Evaluate whether the website is a legitimate news or information source, a known disinformation or propaganda site, a parody/satirical outlet, or an untrustworthy source.
Steps:
1. Identify the domain (e.g., cnn.com, theonion.com, infowars.com).
2. Check for known flags: Listed on misinformation databases, Frequently debunked by fact-checking organizations, Known satire or parody site
3. Provide a concise final judgment and explain your reasoning clearly in 2–4 sentences.
4. Add a description of how a fact-checker should treat the information:
   - Positive use: can be taken as a somewhat reliable or reference-worthy source.
   - Reverse use: if information appears on this website, it is more likely to be false or unreliable; it should be used as evidence for identifying misinformation or propaganda, rather than as factual content.
   -
    """

    prompt = prompt_template + """ ### Output Format:
    {
        "source_identification": "Example.com",
        "type": "<unreliable|satire|reliable|unsure|factcheck>"
        "reasoning":"your judgment and reasoning"
        "fact_checker_usage":"a description of how a fact-checker should treat the information"
    }
    Perform your analysis and provide the output in JSON format based on the given input.
    Here is the web to assess:
    """ + source

    # 生成消息内容，包含图片和文本
    content = [
        {
            "type": "text",
            "text": prompt  # 你原来的文本输入
        }
    ]

    # 构建多个agents的对话信息
    results, total_tokens =  llm_process(content, image_path=None)
    
    return results, total_tokens
  