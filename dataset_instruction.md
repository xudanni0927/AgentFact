# RW-Post Dataset Documentation

## Overview
RW-Post is a multimodal fact-checking dataset consisting of real-world social media posts with aligned image-text content. Each sample includes a claim, post context, and evidence extracted from fact-checking articles, along with reasoning traces for explainable verification.

## Data Structure

Each data instance contains the following fields:

- claim: The textual claim to be verified.
- post_text: The original social media post text.
- image_address: Local path to the associated image.  - image_link: Online URL to the associated image.
- label: Ground-truth veracity label (e.g., TRUE, FALSE, UNPROVEN).
- reasoning_logic: Human-aligned reasoning explaining the verification process.
- key_points: Itemized verification points in the reasoning process.
- fact_checking_evidence: A list of evidence items, including source, content, and URL.
- claim_time: Timestamp of the claim posted on the fact-checking website, which can be used to prevent temporal leakage during retrieval.

Additional metadata may include:
- post_url: Post's source link
- news_url: The source link of Snopes (fact-check) article
- OCR  information (if applicable)

Example structure: 

## Recommended Usage

### Input
Users are recommended to use the following fields as model input:
- claim
- post_text
- image_address

### Output
Models should produce:
- label (veracity prediction)
- reasoning_logic (explanation) [with itemized key point if applicable]
- fact_checking_evidence (supporting evidence)  [mapped to each key point if applicable]

## Evaluation Settings

RW-Post supports multiple evaluation settings:
- Closed-book: Using only claim, post context (image and text)
- Evidence-bounded: Besides claim and post context, using provided ground-truth evidence
- Open-web: Besides claim and post context, retrieving external evidence 
  
## Notes on Data Leakage
To prevent label leakage in the open-web setting, users should:
(1) restrict retrieval to information published before the claim time (using the field `claim_time`), and  
(2) avoid retrieving content from fact-checking websites (e.g., Snopes, PolitiFact, FactCheck.org), which may directly reveal the ground-truth label. Use [is_fact_check.py](./is_fact_check.py) to check whether a given evidence URL is from a fact-checking website.


## Intended Use

This dataset is designed for research on:
- Multimodal fact-checking
- Image-text reasoning
- Evidence grounding
- Explainable AI in multimedia systems

## License and Usage

Please ensure compliance with original data sources (i.e., Snopes websites). The dataset is intended for research purposes only.
