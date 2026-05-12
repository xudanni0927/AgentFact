# ==================================================
# Fact-checking related configuration
# ==================================================
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


