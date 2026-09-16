# Major modifications applied by Technical University of Darmstadt, FG Multimodal Grounded Learning.
# Copyright 2024 Google LLC

"""Class for querying the Google Serper API."""
import random
import time
from datetime import datetime
from typing import Any, Optional

import requests

from config.globals import api_keys
from search.common import Query, WebSource
from agent_utils import get_base_domain

_SERPER_URL = 'https://google.serper.dev'


class SerperAPI:
    """
    Wrapper class for the Google Serper API.
    Handles query execution, retries, and response parsing.
    """

    def __init__(
        self,
        gl: str = 'us',
        hl: str = 'en',
        tbs: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.serper_api_key = api_keys["serper_api_key"]
        self.gl = gl
        self.hl = hl
        self.tbs = tbs

    # ==================================================
    # Public search interface
    # ==================================================
    def search(self, query: Query):
        """
        Execute a search query using the Serper API.

        Args:
            query (Query):
                Search query object containing text, limit,
                optional end date, and optional image flag.

        Returns:
            list[dict]:
                A list of search results, each containing:
                - href
                - title
                - snippet
                - body
        """
        assert self.serper_api_key, 'Missing serper_api_key.'
        assert query is not None, 'Query must not be None.'
        assert query.text, 'Query text must not be None.'

        # ----------------------------------------
        # Build date filter (tbs) if end_date exists
        # ----------------------------------------
        if query.end_date is not None:
            end_date = query.end_date.strftime('%m/%d/%Y')
            tbs = f"cdr:1,cd_min:1/1/1900,cd_max:{end_date}"
        else:
            tbs = self.tbs

        # Determine search type
        search_type = "image" if query.has_image() else "search"

        # Call Serper API
        output = self._call_serper_api(
            query.text,
            gl=self.gl,
            hl=self.hl,
            tbs=tbs,
            search_type=search_type,
        )

        # ----------------------------------------
        # Normalize output format
        # ----------------------------------------
        result_list = []
        for result in output.get("organic", []):
            result_list.append({
                "href": result.get("link", ""),
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                # Use snippet as body fallback
                "body": result.get("snippet", "")
            })

        return result_list

    # ==================================================
    # Low-level API call with retry logic
    # ==================================================
    def _call_serper_api(
        self,
        search_term: str,
        search_type: str = 'search',
        max_retries: int = 20,
        **kwargs: Any,
    ) -> dict[Any, Any]:
        """
        Perform a POST request to the Serper API with retry
        and exponential backoff on timeout.

        Args:
            search_term (str): Search query string.
            search_type (str): 'search' or 'image'.
            max_retries (int): Maximum retry attempts.

        Returns:
            dict: Parsed JSON response from Serper API.
        """
        headers = {
            'X-API-KEY': self.serper_api_key or '',
            'Content-Type': 'application/json',
        }

        params = {
            'q': search_term,
            **{key: value for key, value in kwargs.items() if value is not None},
        }

        response = None
        num_tries = 0
        sleep_time = 0

        while response is None and num_tries < max_retries:
            num_tries += 1
            try:
                response = requests.post(
                    f'{_SERPER_URL}/{search_type}',
                    headers=headers,
                    params=params,
                    timeout=3,
                )

                # Handle credit exhaustion explicitly
                if response.status_code == 400:
                    message = response.json().get('message')
                    if message == "Not enough credits":
                        error_msg = (
                            "No Serper API credits left. "
                            "Please recharge the Serper account."
                        )
                        raise RuntimeError(error_msg)

            except requests.exceptions.Timeout:
                # Exponential backoff with upper bound
                sleep_time = min(sleep_time * 2, 600)
                sleep_time = random.uniform(1, 10) if not sleep_time else sleep_time
                print(
                    f"Unable to reach Serper API: connection timed out. Retrying after {sleep_time:.2f} seconds."
                )
                time.sleep(sleep_time)

        if response is None:
            raise ValueError('Failed to receive a response from Serper API.')

        response.raise_for_status()
        return response.json()

   

    def _parse_sources(self, response: dict, query: Query) -> list[WebSource]:
        """
        Parse web sources from Serper API response and
        convert them into WebSource objects.
        """
        sources = []
        result_key = "images" if query.has_image() else "organic"

        filtered_results = filter_unique_results_by_domain(
            response[result_key]
        )

        if result_key in response:
            for result in filtered_results:
                if len(sources) >= query.limit:
                    break

                url = (
                    result.get("link")
                    if result_key == "organic"
                    else result.get("imageUrl")
                )
                if not url:
                    continue

                title = result.get('title')

                try:
                    result_date = datetime.strptime(
                        result['date'], "%b %d, %Y"
                    ).date()
                except (ValueError, KeyError):
                    result_date = None

                sources.append(
                    WebSource(
                        reference=url,
                        release_date=result_date,
                        title=title
                    )
                )

        return sources


# ==================================================
# Global Serper API instance
# ==================================================
serper_api = SerperAPI()


# ==================================================
# Filter search results by unique base domain
# ==================================================
def filter_unique_results_by_domain(results):
    """
    Filter search results so that only one result per
    base domain is retained (e.g., 'facebook.com'
    regardless of subdomain).

    Args:
        results (list[dict]):
            List of search result dictionaries.

    Returns:
        list[dict]:
            Filtered list containing at most one result
            per base domain.
    """
    unique_domains = set()
    filtered_results = []

    for result in results:
        url = result.get("link", "")
        if not url:
            continue

        base_domain = get_base_domain(url)

        # Retain the first result seen for each domain
        if base_domain not in unique_domains:
            unique_domains.add(base_domain)
            filtered_results.append(result)

    return filtered_results
