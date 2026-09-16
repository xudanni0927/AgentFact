from query_evidence import (
    query_search,
    extract_page_content_and_map_image,
    load_blocked_domains,
    get_domain_from_url
)
import os
import json
import logging
import urllib.parse
from urllib.parse import urlparse
from newspaper import Article, Config
from openai import OpenAI
from configs import prompts_root, OPENAI_KEY, out_root
from google.cloud import vision
from pathlib import Path
# ==================================================
# Logging configuration
# ==================================================
# Suppress verbose logging from newspaper package
for name in ("newspaper", "newspaper.images", "newspaper.network"):
    logging.getLogger(name).setLevel(logging.ERROR)


# ==================================================
# OpenAI client initialization
# ==================================================
client = OpenAI(api_key=OPENAI_KEY)


# ==================================================
# Fact-checking related configuration
# ==================================================
untrusted_sources = {""}

fact_checking_keywords_update = [
    "snopes", "politifact", "factcheck", "truthorfiction", "hoax-slayer",
    "eadstories", "opensecrets", "fullfact", "checkyourfact",
    "mediabiasfactcheck", "poynter", "realitycheck", "fact-check", "apnews",
    "africacheck", "altnews", "boomlive", "factly", "factnameh", "rappler",
    "verafiles", "faktisk", "stopfake", "newtral", "maldita",
    "pagellapolitica", "factcheck.ge", "dubawa", "tsek.ph",
    "dpa-factchecking", "correctiv", "eldiario", "elliberal",
    "fact-or-fiction"
]


# ==================================================
# Utility: check if URL is a known fact-checking site
# ==================================================
def is_fact_checking_site(url: str) -> bool:
    """
    Check whether the given URL belongs to a known fact-checking website.
    Such sites are excluded to avoid circular verification.
    """
    if any(keyword in url.lower() for keyword in fact_checking_keywords_update):
        print(f"url {url} is removed (fact-checking site)")
        return True
    print(f"url {url} is kept")
    return False


# ==================================================
# Source filtering for search results
# ==================================================
def source_filter(results):
    """
    Filter out unsupported or fact-checking sources
    from raw search results.
    """
    if not results:
        return []

    filtered_results = []
    for result in results:
        link = result.get('href', '')
        print("link:", link)
        if is_fact_checking_site(link) or is_unsupported_site(link):
            continue
        filtered_results.append(result)

    return filtered_results


# ==================================================
# Lightweight web scraper using newspaper3k
# ==================================================
def scraper(url, max_len=2000, timeout=5):
    """
    Download and parse article text using newspaper3k
    with a lightweight configuration.
    """
    if not url or not url.startswith(("http://", "https://")):
        return "", "", ""

    # Minimal configuration to reduce overhead
    cfg = Config()
    cfg.request_timeout = timeout
    cfg.fetch_images = False
    cfg.memoize_articles = False
    cfg.keep_article_html = False

    try:
        article = Article(url, config=cfg)
        article.download()
        article.parse()
    except Exception:
        return "", "", ""

    publish_date = article.publish_date or ""
    text = article.text or ""

    body_text = f"publish date: {publish_date}\n\n{text}"
    body_text = body_text[:max_len]

    return body_text, None, article.source_url


# ==================================================
# Text-based search interface
# ==================================================
def text_search(
    query,
    top_k=5,
    retries=3,
    max_date=None
):
    """
    Perform text-based web search and filter sources.
    """
    for attempt in range(retries):
        try:
            results = query_search(query, top_k, max_date=max_date)
            print("results before filter:", results)
            results = source_filter(results)
            print("results after filter:", results)
            return results[:top_k]
        except Exception as e:
            print(f"Error in text_search: {e}")
            continue


# ==================================================
# Evidence extraction from search results
# [Paper] Agent-TR sub-step 3/4: Query-Guided Filtering (Sec. III-C.2).
# Filters/summarizes raw scraped pages down to only the content relevant
# to the input query, via an LLM-based extraction prompt.
# 论文对应：Agent-TR 子步骤(3/4)：按查询过滤（方法节 III-C.2）。用 LLM
# 抽取 prompt，把抓取到的网页原文过滤/总结成只保留与查询相关的内容。
# ==================================================
def evidence_extraction(
    search_results,
    query,
    max_len=2000,
    max_items=3
):
    """
    Extract and summarize evidence from search results
    using LLM-based evidence extraction.
    """
    documents = {}
    headers = {}
    headers_with_link = {}

    for id, search_result in enumerate(search_results):
        title = search_result['title']
        link = search_result['href']
        body = search_result["body"]

        domain = get_domain_from_url(link)
        blocked_domains = load_blocked_domains()

        # Skip blocked domains
        if domain in blocked_domains:
            print(f"Skipped blocked domain: {domain}")
            continue

        # Skip unsupported file types
        parsed_url = urllib.parse.urlparse(link)
        file_ext = os.path.splitext(parsed_url.path)[1].lower()
        unsupported_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '.ppt', '.pptx', '.zip', '.rar'
        }
        if file_ext in unsupported_extensions:
            print(f"Skipped unsupported file type: {file_ext}")
            continue

        full_text, _, source_url = scraper(link, max_len)

        if len(full_text) < 100 or len(full_text) < len(body):
            documents[str(id)] = body
        else:
            documents[str(id)] = full_text

        headers[str(id)] = (
            f"Title: {title}. Source url: + {link}. Article content: "
        )
        headers_with_link[str(id)] = headers[str(id)]

    # Load evidence extraction prompt
    with open(prompts_root + 'evidence_extraction.md', 'r', encoding='utf-8') as f:
        prompt = f.read()

    prompt = prompt.format(
        EVIDENCE=json.dumps(documents),
        TEXT=query
    )

    # Query LLM for evidence extraction
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    response = completion.choices[0].message.content

    evidences = []
    evidences_with_link = []

    try:
        extracted_results = json.loads(response)
        for id, extracted_result in extracted_results.items():
            if extracted_result:
                evidences.append(headers[str(id)] + extracted_result)
                evidences_with_link.append(
                    headers_with_link[str(id)] + extracted_result
                )
    except Exception:
        print(
            "Invalid response from evidence_extraction. \n Fallback to raw documents."
        )
        evidences = list(documents.values())

    evidences = [
        evidence[:max_len]
        for evidence in evidences
        if evidence
    ]

    print("Total relevant evidence:", len(evidences))

    if len(evidences) > max_items:
        return evidences[:max_items], evidences_with_link[:max_items]
    return evidences, evidences_with_link


# ==================================================
# Top-level evidence retrieval pipeline
# [Paper] Agent-TR sub-steps 2-3: Web Search + Query-Guided Filtering
# (Sec. III-C.2). Orchestrates text_search() (Serper web search) and
# evidence_extraction() (LLM-based filtering/summarization) for each
# query generated by Agent-TR's query-generation step.
# 论文对应：Agent-TR 子步骤(2-3)：网页搜索 + 按查询过滤（方法节
# III-C.2）。对 Agent-TR 查询生成步骤产出的每个查询，依次调用
# text_search()（Serper网页搜索）和 evidence_extraction()（LLM过滤总结）。
# ==================================================
def get_evidence(
    text,
    title,
    questions,
    out_root,
    max_len=2000,
    max_date=None
):
    """
    End-to-end pipeline:
    query → search → filter → scrape → extract evidence
    """

    query_set = questions

    all_search_results = {query: [] for query in query_set}
    titles_seen = set()

    # Perform search for each query
    for qid, query in enumerate(query_set):
        results = text_search(query, top_k=10, max_date=max_date)

        for result in results:
            if result['title'] in titles_seen:
                continue
            titles_seen.add(result['title'])
            all_search_results[query].append(result)

    # Logging
    search_log = {
        "original_post": text,
        "title": title,
        "questions": questions,
        "original_search_results": all_search_results
    }

    print("Search log:", search_log)

    # Evidence extraction
    retrieved_dict = {}
    retrieved_dict_with_link = {}
    retrieved_list = []
    retrieved_list_with_link = []

    for question, evidences in all_search_results.items():
        print("Running evidence_extraction...")
        info_list, info_list_with_link = evidence_extraction(evidences, question, max_len)
        retrieved_dict[f"Information related to '{question}'"] = info_list
        retrieved_dict_with_link[f"Information related to '{question}'"] = info_list_with_link

        retrieved_list.extend(info_list)
        retrieved_list_with_link.extend(info_list_with_link)

    search_log["retrieved_text"] = retrieved_dict_with_link

    # Persist search log
    with open(out_root + "search_results.jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps(search_log, ensure_ascii=False, indent=4))

    return (
        retrieved_list,
        retrieved_list_with_link,
        json.dumps(retrieved_dict),
        json.dumps(retrieved_dict_with_link)
    )


# ==================================================
# Domain utilities
# ==================================================
def get_domain(url: str) -> str:
    """
    Extract the registrable domain from a URL
    by removing subdomains.
    Example:
        https://sub.example.com/path → example.com
    """
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()
    domain = '.'.join(netloc.split('.')[-2:])
    return domain


def read_urls_from_file(file_path):
    """
    Read URLs line by line from a text file.
    """
    with open(file_path, 'r') as f:
        return f.read().splitlines()


# ==================================================
# Google Vision API: Web detection from local image
# [Paper] Agent-IR — reverse image search step (Sec. III-C.3; Workflow
# Step 2: Evidence Acquisition). Uses Google Cloud Vision's web detection
# to find visually similar images and the pages that host them.
# 论文对应：Agent-IR 的反向图片搜索步骤（方法节 III-C.3；工作流第2步：
# 证据获取）。调用 Google Cloud Vision 的 web detection 接口，检索视觉
# 相似的图片及其所在网页。
# ==================================================
def get_web_detection_from_local_image(image_path):
    """
    Use Google Cloud Vision API to perform web detection
    on a local image file.

    Returns:
        - full_matches: URLs of fully matching images
        - pages: URLs of pages containing matching images
        - partial_matches: URLs of partially matching images
    """
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    annotations = response.web_detection

    full_matches = [img.url for img in annotations.full_matching_images]
    pages = [page.url for page in annotations.pages_with_matching_images]
    partial_matches = [img.url for img in annotations.partial_matching_images]

    return full_matches, pages, partial_matches


# ==================================================
# Unsupported / unscrapable source configuration
# ==================================================
unsupported_domains_file = Path(__file__).parent / "unsupported_domains.txt"
unsupported_domains = read_urls_from_file(unsupported_domains_file)

unscrapable_urls = [
    "https://www.thelugarcenter.org/ourwork-Bipartisan-Index.html",
    "https://data.news-leader.com/gas-price/",
    "https://www.wlbt.com/2023/03/13/3-years-later-mississippis-/",
    "https://edition.cnn.com/2021/01/11/business/no-fl",
    "https://www.thelugarcenter.org/ourwork-Bipart",
    "https://www.linkedin.com/pulse/senator-kelly-/",
    "http://libres.uncg.edu/ir/list-etd.aspx?styp=ty&bs=master%27s%20thesis&amt=100000",
    "https://www.washingtonpost.com/investigations/coronavirus-testing-denials/2020/03/",
]


def is_unsupported_site(url: str) -> bool:
    """
    Check whether a URL belongs to an unsupported or
    unscrapable website.
    """
    domain = get_domain(url)
    return (
        domain.endswith(".gov")
        or domain in unsupported_domains
        or url in unscrapable_urls
    )


# ==================================================
# Visual search using local image and web detection
# [Paper] Agent-IR — top-level entry point, wraps
# get_web_detection_from_local_image() above with source filtering and
# page-content extraction, producing the evidence items later consumed
# by call_image_similarity_type_and_manipulation_judger /
# call_image_miscaption_detector in agent_model.py.
# 论文对应：Agent-IR 的顶层入口函数，把上面的
# get_web_detection_from_local_image() 与信源过滤、网页正文抓取组装
# 起来，产出的证据项会传给 agent_model.py 里的
# call_image_similarity_type_and_manipulation_judger /
# call_image_miscaption_detector 使用。
# ==================================================
def visual_search(
    source,
    out_root,
    is_url=True,
    max_items=5,
    threshold=0.8
):
    """
    Perform visual search using a local image file.

    Steps:
    1. Run Google Vision web detection
    2. Filter detected pages
    3. Extract page content and map images
    4. Save search logs
    """

    # If the source is already a URL, skip visual search
    if is_url:
        return None

    # ----------------------------------------
    # Prepare local image
    # ----------------------------------------
    image_path = os.path.abspath(source)

    # Run Google Vision web detection
    full_matches, pages, partial_matches = \
        get_web_detection_from_local_image(image_path)

    # Collect page URLs containing matching images
    search_results_pages = []
    for page_url in pages:
        search_results_pages.append({
            "href": page_url
        })

    # Merge full and partial image matches
    full_matches = full_matches + partial_matches

    # ----------------------------------------
    # Source filtering
    # ----------------------------------------
    source_filtered_results = source_filter(search_results_pages)

    search_log = {
        "input_image": source,
        "original_search_results_all": search_results_pages,
        "source_filtered_results": source_filtered_results
    }

    # ----------------------------------------
    # Extract page content and map images
    # ----------------------------------------
    return_list = []
    result_count = 0

    for search_result in source_filtered_results:
        if result_count >= max_items:
            break

        page_result = extract_page_content_and_map_image(
            search_result['href'],
            full_matches
        )

        if page_result:
            return_list.append(page_result)
            result_count += 1

    search_log["duplicate_image_results"] = return_list

    # Persist visual search logs
    with open(out_root + "visual_search_results.jsonl", 'a', encoding='utf-8') as f:
        f.write(json.dumps(search_log, ensure_ascii=False))
        f.write("\n")

    return return_list


# ==================================================
# Selenium driver cleanup (if used)
# ==================================================
def driver_quit():
    """
    Quit the global Selenium WebDriver instance.
    """
    global driver
    driver.quit()
