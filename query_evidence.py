import requests
from bs4 import BeautifulSoup
import random
import time
import os
import re
from urllib.parse import urlparse
from search.serper import serper_api
from search.common import Query
from configs import API_KEY, CSE_ID


def google_search(
    query,
    api_key,
    cse_id,
    num_results=5,
    site="",
    max_date=None
):
    """
    Perform web search using Google Custom Search JSON API.

    Args:
        query (str):
            Search query string.
        api_key (str):
            Google API key.
        cse_id (str):
            Custom Search Engine (CSE) ID.
        num_results (int):
            Number of search results to return (max 10).
        site (list or str):
            Optional list of site URLs/domains to restrict search scope.
        max_date (str):
            Optional date constraint (format depends on Google syntax).

    Returns:
        list[dict]:
            A list of search results, each containing:
            - title
            - link
            - snippet

    Raises:
        RuntimeError:
            If all retry attempts fail due to request errors
            or non-200 HTTP responses.
    """

    url = "https://www.googleapis.com/customsearch/v1"

    # ----------------------------------------
    # Construct site-restricted query if needed
    # ----------------------------------------
    if site:
        # Normalize site list to domain-only format
        site = [
            re.sub(r"https?://(www\.)?", "", site_i).strip("/")
            for site_i in site
        ]

        # Combine sites using Google 'site:' syntax
        site_query = " OR ".join(f"site:{site_i}" for site_i in site)

        # Append site filter to the original query
        query = f"{site_query} {query}"

    # ----------------------------------------
    # Append date constraint if provided
    # ----------------------------------------
    if max_date:
        query += f" before:{max_date}"

    # ----------------------------------------
    # Request parameters
    # ----------------------------------------
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    # ----------------------------------------
    # Retry logic
    # ----------------------------------------
    attempt = 0
    fail_count = 0

    while attempt < 3:
        try:
            start_time = time.time()

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=5
            )

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Request execution time: {execution_time:.4f} seconds")

            if response.status_code == 200:
                results = response.json().get("items", [])
                return [
                    {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet")
                    }
                    for item in results
                ]

            else:
                print(
                    f"HTTP error: {response.status_code} - "
                    f"{response.text}"
                )
                fail_count += 1
                attempt += 1

        except Exception as e:
            print("Request exception:", e)
            attempt += 1

    # ----------------------------------------
    # All retry attempts failed
    # ----------------------------------------
    if fail_count == 3:
        raise RuntimeError(
            "All three Google search requests failed "
            "(HTTP errors or exceptions)."
        )

 
BLOCKLIST_FILE = 'blocked_domains.txt'


# ==================================================
# Blocked domain utilities
# ==================================================
def load_blocked_domains():
    """
    Load blocked domains from a local file.

    Returns:
        set[str]: A set of blocked domain names.
    """
    if not os.path.exists(BLOCKLIST_FILE):
        return set()

    with open(BLOCKLIST_FILE, 'r', encoding='utf-8') as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }


def save_blocked_domain(domain):
    """
    Append a domain to the blocked domains file.
    """
    with open(BLOCKLIST_FILE, 'a', encoding='utf-8') as f:
        f.write(domain + '\n')


def get_domain_from_url(url):
    """
    Extract the domain (netloc) from a URL.
    """
    return urlparse(url).netloc.lower()


# ==================================================
# Page content extraction with image mapping
# ==================================================
def extract_page_content_and_map_image(url, images_match_urls):
    """
    Fetch and parse webpage content, including title and body text,
    while minimizing the risk of request blocking.

    Args:
        url (str): Webpage URL.
        images_match_urls (list): URLs of images matched via visual search
                                  (kept for interface consistency).

    Returns:
        dict or None:
            {
                "href": url,
                "title": page title,
                "body": cleaned text content
            }
            Returns None if the page cannot be fetched.
    """

    domain = get_domain_from_url(url)
    blocked_domains = load_blocked_domains()

    # Skip blocked domains
    if domain in blocked_domains:
        print(f"⚠️ Skipped blocked domain: {domain}")
        return None

    # Randomized headers to mimic different browsers
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0",
        ]),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        # Random delay to simulate human browsing behavior
        time.sleep(random.uniform(0.5, 1.5))

        response = requests.get(url, headers=headers, timeout=3)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Extract page title
        title = soup.title.string.strip() if soup.title else ""

        # Extract and clean paragraph text
        paragraphs = soup.find_all("p")
        raw_text = "\n".join(
            p.get_text(strip=True)
            for p in paragraphs
            if p.get_text(strip=True)
        )

        # Remove non-ASCII characters and normalize whitespace
        text_content = re.sub(
            r'\s+',
            ' ',
            re.sub(r'[^\x00-\x7F]+', ' ', raw_text)
        )

        # Limit text length
        text_content = (
            text_content
            if len(text_content) < 1000
            else text_content[:1000]
        )

        return {
            "href": url,
            "title": title,
            "body": text_content
        }

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch content from {url}: {e}")
        save_blocked_domain(domain)
        return None


# ==================================================
# Query-based web search wrapper
# ==================================================
def query_search(query, num_results=5, max_date=None):
    """
    Perform web search using the configured search API.

    Args:
        query (str): Search query.
        num_results (int): Number of results to return.
        max_date (str): Optional end date constraint.

    Returns:
        list[dict]: Search results with basic metadata.
    """
    search_api = "serper"

    if search_api == "GOOGLE_CUSTOM_API":
        search_results = google_search(
            query,
            API_KEY,
            CSE_ID,
            num_results,
            max_date=max_date
        )
        contents = []

        for result in search_results:
            content = {
                "href": result['link'],
                "title": result['title'],
                "snippet": result["snippet"],
                "body": result["snippet"]
            }

            if content and len(content['body']) > 10:
                contents.append(content)

    elif search_api == "serper":
        print("serper is running")
        query_s = Query(
            text=query,
            limit=num_results,
            end_date=max_date
        )
        contents = serper_api.search(query_s)

    return contents



