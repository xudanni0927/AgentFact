import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

prompts_root = "prompts/"
out_root = "output/"
cache_root = "data/cache/"
if not os.path.exists(cache_root):
    os.makedirs(cache_root)

definition_path = prompts_root + "definition.json"

# Set these via environment variables (e.g. a .env file, see .env.example)
# rather than editing this file, so real keys never get committed.
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# Used for Google Custom Search (only needed if search_api == "GOOGLE_CUSTOM_API"
# in query_evidence.py; the default "serper" backend uses config/api_keys.yaml instead)
API_KEY = os.environ.get("GOOGLE_CSE_API_KEY", "")
CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")
