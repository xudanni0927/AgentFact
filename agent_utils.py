import os
import openai
from PIL import Image
from io import BytesIO
import json
import time
import requests
from PIL import Image
import base64
from openai import OpenAI
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

model_version = 'gpt-4o-mini'
# model_version= "llava"
# set your API key via environment variables (see .env.example), not here
if "gpt" in model_version:
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = "https://api.openai.com/v1/chat/completions"
    response_format= "json"
    client = OpenAI(
    # This is the default and can be omitted
    api_key=openai.api_key)
if "deepseek" in model_version:
    openai.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url="https://api.deepseek.com/chat/completions"
    response_format= { "type": "json_object" }
    client = OpenAI(
    # This is the default and can be omitted
    api_key=openai.api_key)

if model_version == "llava":
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path

    model_path = "liuhaotian/llava-v1.6-vicuna-7b"

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        get_model_name_from_path(model_path),
        load_in_8bit=True,
        device_map="auto"
    )

    model.eval()

    image = Image.open("tampered.jpg").convert("RGB")

    prompt = (
        "Analyze whether this image shows signs of manipulation. "
        "If so, describe the suspicious regions."
    )

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            images=image_tensor,
            max_new_tokens=256
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))

if model_version == "llama":
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.2-8B",
        messages=[{"role": "user", "content": "Plan a task"}],
    )
    print(resp.choices[0].message.content)



from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages, image_path=None, max_new_tokens=512):
        """
        messages: List[{"role": "user"/"assistant", "content": str}]
        image_path: str or None
        return: str (raw text output)
        """
        pass

class OpenAIWrapper(BaseLLM):
    def __init__(self, model_name, base_url, api_key):
        self.model_name = model_name
        self.base_url = base_url
        openai.api_key = api_key

    def generate(self, messages, image_path=None, max_new_tokens=2000):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens
        }

        # 如果是多模态 GPT（如 gpt-4o）
        if image_path is not None:
            payload["messages"] = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": messages[-1]["content"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"file://{image_path}"}
                    }
                ]
            }]
        else:
            payload["messages"] = [{
                "role": "user",
                "content": messages
            }]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        response = requests.post(self.base_url, headers=headers, json=payload)
        

        
        response.raise_for_status()

        return response.json()
    

if model_version=="llava":
    from llava.conversation import conv_templates
    from PIL import Image
    import torch

    class LLaVAWrapper(BaseLLM):
        def __init__(self, tokenizer, model, image_processor):
            self.tokenizer = tokenizer
            self.model = model
            self.image_processor = image_processor

        def generate(self, messages, image_path=None, max_new_tokens=512):
            # 1. messages → 纯文本 prompt
            text_prompt = ""
            for msg in messages:
                role = msg["role"].capitalize()
                text_prompt += f"{role}: {msg['content']}\n"

            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            image_tensor = None
            if image_path:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.image_processor(
                    image, return_tensors="pt"
                )["pixel_values"].to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    images=image_tensor,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image
    import torch

    class Qwen2VLWrapper(BaseLLM):
        def __init__(self, model_id):
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True
            )
            self.model.eval()

        def generate(self, messages, image_path=None, max_new_tokens=512):
            content = []

            if image_path:
                content.append({"type": "image"})

            content.append({
                "type": "text",
                "text": messages[-1]["content"]
            })

            inputs = self.processor(
                [{"role": "user", "content": content}],
                images=Image.open(image_path).convert("RGB") if image_path else None,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens
                )

            return self.processor.decode(output_ids[0], skip_special_tokens=True)

# llm = Qwen2VLWrapper("Qwen/Qwen2-VL-7B-Instruct")
# or
    llm = LLaVAWrapper(tokenizer, model, image_processor)
# or
elif model_version in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]:
    llm = OpenAIWrapper(model_version, base_url, openai.api_key)

def gpt_agent_review_reason(content, prompt, agents, rounds, module_name, output_dir):
    start_time = time.time()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    agent_contexts = [[{
                "role": "user",
                "content": content
            }] for agent in range(agents)]
    agent_contexts_answer =[[{
                "role": "user",
                "content": prompt
            }]for agent in range(agents)]
    
    for round in range(rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]

                if round == (rounds - 1):
                    message = construct_message_review_reason(agent_contexts_other, 2*round - 1, final=True)
                else:
                    message = construct_message_review_reason(agent_contexts_other, 2*round - 1, final=False)
                agent_context.append(message)
                agent_contexts_answer[i].append(message)
            answer = None

            # 构建payload 
            if "deepseek" in model_version:
                payload = {
                    "model": model_version,
                    "messages": agent_context,
                    "response_format": response_format,
                    "max_tokens": 4000
                }
            else:
                payload = {
                    "model": model_version,
                    "messages": agent_context,
                    "max_tokens": 4000
                }  
            try:                  
                completion = requests.post(base_url, headers=headers, json=payload)
                answer = completion
            except:
                try:                   
                    completion = requests.post(base_url, headers=headers, json=payload)
                    answer = completion
                except Exception as e:
                    print(f"获取gpt响应时发生错误: {e}")

            if answer:
                assistant_message = construct_assistant_message(completion.json())

                agent_context.append(assistant_message)
                agent_contexts_answer[i].append(assistant_message)

            if round==(rounds - 1):
                break


    end_time = time.time()  # 记录结束时间
    success_log = output_dir + module_name + '_success_log.jsonl'
    fail_log = output_dir + module_name + '_fail_log.jsonl'
    #解析agent_debate得到的结果
    results = {}
    post_results = {}
    response_data = completion.json()
    # 提取 content 字段
    outputs = response_data['choices'][0]['message']['content']
    total_tokens = response_data["usage"]["total_tokens"]
    # outputs = response.choices[0].message.content 
    total_tokens = 0
    # outputs = response.choices[0].message.content 
    if len(outputs) > 0:
        total_tokens = response_data["usage"]["total_tokens"]
        state, results = convert_to_json(outputs)
        if (state):
            with open(success_log, 'a') as f:
                json.dump(agent_contexts_answer[0], f, indent=4, ensure_ascii=False)
                f.write('\n')
            if rounds == 1:
                post_results["updated_answer"] = results
                post_results["time_seconds"]= f"{end_time - start_time:.2f}"
                return post_results, total_tokens
            results["time_seconds"]= f"{end_time - start_time:.2f}"
            return results, total_tokens
    else:
        print("does not have result")
    with open(fail_log, 'a') as f:
        json.dump(agent_contexts_answer[0], f, indent=4, ensure_ascii=False)
        f.write('\n')
    return results, 0

def gpt_agent_review_query(content, prompt, agents, rounds, module_name, output_dir):
    start_time = time.time()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    agent_contexts = [[{
                "role": "user",
                "content": content
            }] for agent in range(agents)]
    agent_contexts_answer =[[{
                "role": "user",
                "content": prompt
            }]for agent in range(agents)]
    
    for round in range(rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]

                if round == (rounds - 1):
                    message = construct_message_review_queries(agent_contexts_other, 2*round - 1, final=True)
                else:
                    message = construct_message_review_queries(agent_contexts_other, 2*round - 1, final=False)
                agent_context.append(message)
                agent_contexts_answer[i].append(message)
            answer = None
            # 构建payload
            if "deepseek" in model_version:
                payload = {
                    "model": model_version,
                    "messages": agent_context,
                    "response_format": response_format,
                    "max_tokens": 4000
                }
            else:
                payload = {
                    "model": model_version,
                    "messages": agent_context,
                    "max_tokens": 4000
                }  
            try:
                completion = requests.post(base_url, headers=headers, json=payload)
                answer = completion
            except:
                try:
                    completion = requests.post(base_url, headers=headers, json=payload)
                    answer = completion
                except Exception as e:
                    print(f"获取gpt响应时发生错误: {e}")

            if answer:
                assistant_message = construct_assistant_message(completion.json())

                agent_context.append(assistant_message)
                agent_contexts_answer[i].append(assistant_message)

            if round==(rounds - 1):
                break


    end_time = time.time()  # 记录结束时间
    success_log = output_dir + module_name + '_success_log.jsonl'
    fail_log = output_dir + module_name + '_fail_log.jsonl'
    #解析agent_debate得到的结果
    results = {}
    post_results = {}
    response_data = completion.json()
    # 提取 content 字段
    outputs = response_data['choices'][0]['message']['content']
    # outputs = response.choices[0].message.content 
    total_tokens = 0
    # outputs = response.choices[0].message.content 
    if len(outputs) > 0:
        total_tokens = response_data["usage"]["total_tokens"]
        state, results = convert_to_json(outputs)
        if (state):
            with open(success_log, 'a') as f:
                json.dump(agent_contexts_answer[0], f, indent=4, ensure_ascii=False)
                f.write('\n')
            if rounds == 1:
                post_results["updated_answer"] = results
                post_results["time_seconds"]= f"{end_time - start_time:.2f}"
                return post_results, total_tokens
            results["time_seconds"]= f"{end_time - start_time:.2f}"
            return results, total_tokens
    else:
        print("does not have result")
    with open(fail_log, 'a') as f:
        json.dump(agent_contexts_answer[0], f, indent=4, ensure_ascii=False)
        f.write('\n')
    return results, 0

def gpt_agent_review_plan(content, prompt, agents, rounds, module_name, output_dir):
    start_time = time.time()
    agent_contexts = [[{
                "role": "user",
                "content": content
            }] for agent in range(agents)]
    agent_contexts_answer =[[{
                "role": "user",
                "content": prompt
            }]for agent in range(agents)]
    
    for round in range(rounds):
        for i, agent_context in enumerate(agent_contexts):
            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]

                if round == (rounds - 1):
                    message = construct_message_review_plan(agent_contexts_other, 2*round - 1, final=True)
                else:
                    message = construct_message_review_plan(agent_contexts_other, 2*round - 1, final=False)
                agent_context.append(message)
                agent_contexts_answer[i].append(message)
            answer = None
            try:
                answer = llm.generate(agent_context, image_path=None)
            except Exception as e:
                print(f"获取gpt响应时发生错误: {e}")
                try: 
                    answer = llm.generate(agent_context, image_path=None)
                except Exception as e:
                    print(f"获取gpt响应时发生错误: {e}")

            if answer:
                assistant_message = construct_assistant_message(answer.json())

                agent_context.append(assistant_message)
                agent_contexts_answer[i].append(assistant_message)

            if round==(rounds - 1):
                break


    end_time = time.time()  # 记录结束时间
    success_log = output_dir + module_name + '_success_log.jsonl'
    fail_log = output_dir + module_name + '_fail_log.jsonl'
    #解析agent_debate得到的结果
    results = {}
    post_results = {}
    response_data = answer.json()
    # 提取 content 字段
    outputs = response_data['choices'][0]['message']['content']
    total_tokens = 0
    # outputs = response.choices[0].message.content 
    if len(outputs) > 0:
        total_tokens = response_data["usage"]["total_tokens"]
        state, results = convert_to_json(outputs)
        if (state):
            with open(success_log, 'a') as f:
                json.dump(agent_contexts_answer[0], f, indent=4, ensure_ascii=False)
                f.write('\n')
            if rounds == 1:
                post_results["updated_answer"] = results
                post_results["time_seconds"]= f"{end_time - start_time:.2f}"
                return post_results, total_tokens
            results["time_seconds"]= f"{end_time - start_time:.2f}"
            return results, total_tokens
    else:
        print("does not have result")
    with open(fail_log, 'a') as f:
        json.dump(agent_contexts_answer[0], f, indent=4, ensure_ascii=False)
        f.write('\n')
    return results, total_tokens

def llm_process(content, image_path=None):
    start_time = time.time()
    try:
        response_data =llm.generate(content, image_path=None)
    except Exception as e:
        print(f"获取 gpt 响应失败: {e}")
        return {}, 0
    end_time = time.time()

    outputs = response_data['choices'][0]['message']['content']
    total_tokens = response_data["usage"]["total_tokens"]

    if outputs:
        state, results = convert_to_json(outputs)
        if state:
            return {
                "updated_answer": results,
                "time_seconds": f"{end_time - start_time:.2f}"
            }, total_tokens
    else:
        print("llm does not have result for prompt:", content)

    return {}, 0


def construct_message(agents, idx, final=False):
    prefix_string = "Here are some answers given by other agents: "

    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        response = "\n\n Agent response: ```{}```".format(agent_response)
        if isinstance(response, dict):
            prefix_string = prefix_string + response.get("updated_answer","")
        else:
            prefix_string = prefix_string + str(response)

    if final:
        prefix_string = prefix_string + """\n\n Closely examine your answer and the answer of other agents and provide an updated answer. Also, specify the type of influence their answers had on your update by selecting one of the following categories: 
            1. "Corrected my answer"
            2. "Refined my answer"
            3. "Completed my answer"
            4. "Revised due to conflicting views"
            5. "Confirmed and unchanged"
            6. "Uninfluenced by other answers"). Example output format (json): 
            {
                "updated_answer":"XXX",
                "influence_type": "Completed my answer"
                } 
                """
    else:
        prefix_string = prefix_string + """\n\n Using these other answers as additional advice, what is your updated anwswer. Also, specify the type of influence their answers had on your update by selecting one of the following categories: 
            1. "Corrected my answer"
            2. "Refined my answer"
            3. "Completed my answer"
            4. "Revised due to conflicting views"
            5. "Confirmed and unchanged"
            6. "Uninfluenced by other answers"). Example output format (json): 
            {
                "updated_answer":"XXX",
                "influence_type": "Completed my answer"
                }  """

    return {"role": "user", "content": prefix_string}

def construct_message_review_queries(agents, idx, final=False):
    prefix_string = "Here are some answers given by other agents: "

    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        response = "\n\n Agent response: ```{}```".format(agent_response)
        if isinstance(response, dict):
            prefix_string = prefix_string + response.get("updated_answer","")
        else:
            prefix_string = prefix_string + str(response)

    prefix_string = prefix_string + """\n\n Please evaluate the generated queries above and determine whether they would effectively retrieve the intended information when used directly in the Google search engine. Evaluate whether they basically repeat any query in the previous queries list or each other. Additionally, suggest more suitable query formulations for better search accuracy. Answer with your detailed review and an updated answer: 
    Example output format (json): 
    {
    "review": "Your detailed evaluation of the original queries goes here.",
    "updated_answer": "Your reformulated queries should go here, strictly following the output format of the original agent's answer."
    }
     """
    return {"role": "user", "content": prefix_string}

def construct_message_review_reason(agents, idx, final=False):
    prefix_string = "Here are some answers given by other agents: "

    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        response = "\n\n Agent response: ```{}```".format(agent_response)
        if isinstance(response, dict):
            prefix_string = prefix_string + response.get("updated_answer","")
        else:
            prefix_string = prefix_string + str(response)

    prefix_string = prefix_string + """\n\n Please evaluate the reasoning process above and determine whether each input information is correctly understood and used in the reasoning process. Additionally, suggest more accurate and cautious reasoning. Answer with your detailed review and an updated answer: 
    Example output format (json): 
    {
    "review": "Your detailed evaluation of the original queries goes here.",
    "updated_answer": "Your reformulated reasoning should go here, strictly following the output format of the original agent's answer."
    }
            """
    return {"role": "user", "content": prefix_string}

def construct_message_review_plan(agents, idx, final=False):
    prefix_string = "Here are some answers given by other agents: "

    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        response = "\n\n Agent response: ```{}```".format(agent_response)
        if isinstance(response, dict):
            prefix_string = prefix_string + response.get("updated_answer","")
        else:
            prefix_string = prefix_string + str(response)

    prefix_string = prefix_string + """\n\n Please evaluate the generated plan above for its effectiveness and comprehensiveness in verifying the authenticity of the claim. Specifically:
    1. Does the plan thoroughly cover all necessary aspects of validation without redundancy?
    2. Are individual steps clear and manageable, or do any of them attempt to handle multiple tasks at once?
    3. Is there any overlap or repetition between the "information list" and the "validation list," or within either list itself?

    Based on your evaluation, please provide:
    - A detailed critique of the current plan
    - An improved version of the plan, optimized for clarity, completeness, and efficiency
    Example output format (json): 
    {
    "review": "Your detailed evaluation of the original queries goes here.",
    "updated_answer": "Your improved plan should go here, strictly following the output format of the original agent's answer."
    }
    """
    return {"role": "user", "content": prefix_string}

def construct_message_review(agents, idx, final=False):
    prefix_string = "Here are some answers given by other agents: "

    for i, agent in enumerate(agents):
        agent_response = agent[idx]["content"]
        response = "\n\n Agent response: ```{}```".format(agent_response)
        if isinstance(response, dict):
            prefix_string = prefix_string + response.get("updated_answer","")
        else:
            prefix_string = prefix_string + str(response)

    prefix_string = prefix_string + """\n\n Please evaluate the answer provided by another agent.
    Based on your evaluation, please provide:
    - A detailed critique of the current answer.
    - An improved version of the answer.
    Example output format (json): 
    {
    "review": "Your detailed evaluation of the original queries goes here.",
    "updated_answer": "Your reformulated answer should go here, strictly following the output format of the original agent's answer."
    }
    """
    return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion):
    content = completion['choices'][0]['message']['content']
    return {"role": "assistant", "content": content}

def is_json(myjson):
    if isinstance(myjson, (dict, list)):  # 如果已经是字典或列表，则直接认为是有效的JSON
        return True
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True

def convert_to_json(gpt_output):
    if is_json(gpt_output):
        if isinstance(gpt_output, str):
            gpt_data = json.loads(gpt_output)  # 如果是字符串，则将其转换为字典
        else:
            gpt_data = gpt_output  # 如果已经是字典，则直接使用
        return True, gpt_data
    else:
        # 将非JSON格式的gpt_output发送给GPT进行转换
        try_i = 0
        while try_i < 3:
            converted_output = gpt_convert_to_json(gpt_output)
            try:
                gpt_data = json.loads(converted_output)
                return True, gpt_data
            except json.JSONDecodeError:
                print("Unable to convert output to JSON format.")
                try_i += 1
    
    return False, None

def gpt_convert_to_json(gpt_output):
    # 这里你可以调用GPT模型来转换非JSON格式的gpt_output为JSON格式
    # 此处为示例，假设你有一个函数convert_to_json_using_gpt完成此任务
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用支持对话的模型
        # messages=[
        #     {"role": "user", "content": f"Convert the following text to strict JSON format, ensuring proper handling of special characters (e.g., escape double quotes (\") and other reserved characters) Note, the result has to be strict JSON format: {gpt_output}"}
        # ],
        messages=[
            {"role": "user", "content": (
            "Please help me validate and clean the following JSON string so that it can be successfully parsed by json.loads().\n\n"
            "Specifically, please do the following:\n"
            "1. Remove all code block markers (such as ```json and ```).\n"
            "2. Ensure that all strings are enclosed in standard double quotes and that no non-standard quotes (e.g., smart quotes) are used.\n"
            "3. Verify that the JSON syntax is correct and that the content is a valid JSON object.\n"
            "4. Output only the cleaned, standard JSON content without any additional commentary or formatting instructions.\n"
            "5. Assume the output will be saved with UTF-8 encoding.\n\n"
            f"JSON to process:\n{gpt_output}"
        )}
        ],

        # Specifically, please do the following: 1. Remove all code block markers (such as ```json and ```). 2. Ensure that all strings are enclosed in standard double quotes and that no non-standard quotes (e.g., smart quotes) are used. 3. Verify that the JSON syntax is correct and that the content is a valid JSON object. 4. Output only the cleaned, standard JSON content without any additional commentary or formatting instructions. 5. Assume the output will be saved with UTF-8 encoding. 
        max_tokens=4000
    )
    return response.choices[0].message.content.strip()


def image_to_code(image):
    init_image = Image.open(image)
    if init_image.size[0] > 1024:
        new_width = 1024
        new_height = int((new_width / init_image.size[0]) * init_image.size[1])
        img_resized = init_image.resize((new_width, new_height))
        # 将图像转换为字节流
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")  # 根据需要选择格式，如 "PNG" 或 "JPEG"
        img_bytes = buffered.getvalue()
        code_image = base64.b64encode(img_bytes).decode('utf-8')
    else: 
        # 将图像转换为字节流
        buffered = BytesIO()
        init_image.save(buffered, format="PNG")  # 根据需要选择格式，如 "PNG" 或 "JPEG"
        img_bytes = buffered.getvalue()
        code_image = base64.b64encode(img_bytes).decode('utf-8')
    return code_image

def extract_link(text: str) -> str:
    """
    提取 'Source url: +' 和 'Article content:' 之间的内容
    """
    pattern = r"Source url:\s*\+\s*(.*?)\s*Article content:"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def safe_parse_updated_answer(output, caller_name=""):
    """
    Safely parse `updated_answer` from LLM outputs.
    Handles:
    - dict (already parsed)
    - JSON string
    - malformed JSON fixed via convert_to_json
    """
    updated = output.get("updated_answer", {})

    if isinstance(updated, dict):
        return updated

    if isinstance(updated, str):
        try:
            return json.loads(updated)
        except Exception:
            try:
                _, fixed = convert_to_json(updated)
                print(f"[Fix JSON] {caller_name} output fixed by GPT.")
                return fixed
            except Exception:
                print(f"[Fail JSON] {caller_name} output cannot be fixed:", updated)

    return {}

from urllib.parse import urlparse


def extract_never_used_ids(json_data, evidence_list):
    """
    Traverse the full validation JSON result and collect all evidence IDs
    that appear in the input evidence list but are NEVER marked as used
    in any reasoning or fact-checking step.

    Args:
        json_data (dict):
            Full model output containing validation_result,
            reasoning_steps, and optional fact-check evidence.
        evidence_list (list):
            Original list of evidence items, each containing a unique "id".

    Returns:
        list:
            Evidence IDs that never appear as "is_used == True"
            in any reasoning step or fact-check evidence.
    """

    # Collect all evidence IDs from the original evidence list
    all_ids = {
        evidence.get("id")
        for evidence in evidence_list
        if "id" in evidence
    }

    # Track evidence IDs that are actually used
    used_ids = set()

    def mark_used_ids(evidence_items):
        """
        Helper function to extract and mark used evidence IDs
        from a list of evidence references.

        An evidence item is considered 'used' if:
        - is_used is boolean True
        - OR is_used is a string containing 'true' (case-insensitive)
        """
        for item in evidence_items:
            item_id = item.get("id")
            is_used = item.get("is_used")

            if is_used is True:
                used_ids.add(item_id)
            elif isinstance(is_used, str) and "true" in is_used.lower():
                used_ids.add(item_id)
            # Explicit False or missing values are ignored

    # ----------------------------
    # Process reasoning steps
    # ----------------------------
    reasoning_steps = (
        json_data
        .get("validation_result", {})
        .get("reasoning_steps", [])
    )

    for step in reasoning_steps:
        evidence_items = step.get("relevant_text_evidence_list", [])
        if isinstance(evidence_items, list):
            mark_used_ids(evidence_items)

    # ----------------------------
    # Process direct fact-check evidence (fallback supported)
    # ----------------------------
    validation_result = json_data.get("validation_result", {})

    direct_fact_check_evidence = (
        validation_result.get("direct_fact_check_evidence")
        or validation_result.get("fact_check_evidence")
        or {}
    )

    evidence_items = direct_fact_check_evidence.get(
        "relevant_text_evidence_list", []
    )

    if isinstance(evidence_items, list):
        mark_used_ids(evidence_items)

    # ----------------------------
    # Compute unused evidence IDs
    # ----------------------------
    never_used_ids = list(all_ids - used_ids)
    return never_used_ids


def get_base_domain(url) -> str:
    """
    Extracts the base domain from a given URL, ignoring common subdomains like 'www' and 'm'.

    Args:
        url (str): The URL to extract the base domain from.

    Returns:
        str: The base domain (e.g., 'facebook.com').
    """
    netloc = urlparse(url).netloc

    # Remove common subdomains like 'www.' and 'm.'
    if netloc.startswith('www.') or netloc.startswith('m.'):
        netloc = netloc.split('.', 1)[1]  # Remove the first part (e.g., 'www.', 'm.')

    return netloc
