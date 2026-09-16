You are given a Query. You are then given a dictionary called Documents, whose key is the document ID and value is the documen retrieved from the Internet. For each document, 
- if the article is relevant to Query, summarize the core information comprehensively and concisely
- if it is irrelevant to Query, return empty string.
Please output a new dictionary, whose key is still document ID and value is the document segments relevant to the Query. The length of each segment should be longer than 100 words (if avalible) but shorter than 500 words. List the core facts in itemized points. Do not be strict (think it carefully before you decide a document is irrelevant to the query).

### output format
{{"0":"XXX", "1":"XXX","2": "XXX","3":"XXX"}}

### Your turn

**Query**
{TEXT}

**Documents**
{EVIDENCE}

**Output: (Don't output anything else except for the JSON object. Don't add Markdown syntax like ```json):**
