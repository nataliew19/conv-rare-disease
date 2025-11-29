# Wikipedia Domain Ban - Coded Implementation

## Overview

This document describes the **coded implementation** (not just natural language specification) of the Wikipedia domain ban in the evaluation system. Wikipedia sources are explicitly banned from being used as citations in generated articles.

## Implementation Details

### 1. Domain Detection Function (`is_wikipedia_url`)

**Location:** `eval/eval_article_content.py` and `eval/citation_verifier.py`

A coded function that checks if a URL belongs to Wikipedia or any Wikimedia project:

```python
def is_wikipedia_url(url: str) -> bool:
    """
    Check if a URL belongs to Wikipedia or any Wikimedia project.
    This is a coded check, not just natural language specification.
    """
```

**Banned Domains:**
- `wikipedia.org` (all language subdomains)
- `wikipedia.com`
- `wikimedia.org`
- `wikidata.org`
- `wiktionary.org`
- `wikiquote.org`
- `wikibooks.org`
- `wikisource.org`
- `wikinews.org`
- `wikiversity.org`
- `wikivoyage.org`
- `mediawiki.org`
- `foundation.wikimedia.org`

**Detection Method:**
- Parses URL using `urllib.parse.urlparse`
- Checks exact domain matches
- Checks subdomain matches
- Checks for 'wikipedia' or 'wikimedia' in domain name
- Fallback: checks URL string for 'wikipedia' or 'wiki' keywords

### 2. URL Filtering Functions

**Location:** `eval/eval_article_content.py`

- `filter_wikipedia_urls(urls)`: Filters out Wikipedia URLs from a list
- `extract_urls_from_text(text)`: Extracts all URLs from text
- `check_text_for_wikipedia_sources(text)`: Checks text and returns statistics

### 3. Citation Mapping Filtering

**Location:** `eval/citation_verifier.py`

The `create_citation_mapping_from_sources()` function automatically excludes Wikipedia sources:

```python
def create_citation_mapping_from_sources(sources: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Create a mapping from citation numbers to source passages.
    Wikipedia sources are BANNED and will be excluded.
    """
```

**Filtering Logic:**
1. Checks `url` field in source dictionary
2. Checks URLs embedded in `content`/`text` fields
3. Removes any sources with Wikipedia URLs
4. Logs warnings for removed sources

### 4. Evaluation Integration

**Location:** `eval/eval_article_content.py` (main evaluation script)

When evaluating a generated article:
1. Extracts all URLs from the article text
2. Checks each URL against Wikipedia domain list
3. Reports warnings if Wikipedia sources are found
4. Excludes Wikipedia URLs from citation verification

## Usage

### Checking for Wikipedia Sources

```python
from eval_article_content import check_text_for_wikipedia_sources

# Check generated article
stats = check_text_for_wikipedia_sources(generated_article_text)

if stats['has_wikipedia_sources']:
    print(f"Found {stats['wikipedia_url_count']} Wikipedia URL(s)")
    for url in stats['wikipedia_urls']:
        print(f"  - {url}")
```

### Filtering URLs

```python
from eval_article_content import filter_wikipedia_urls

urls = ["https://example.com", "https://en.wikipedia.org/wiki/DMD"]
filtered, removed = filter_wikipedia_urls(urls)
# filtered: ["https://example.com"]
# removed: ["https://en.wikipedia.org/wiki/DMD"]
```

### Creating Citation Mapping (Auto-filtered)

```python
from citation_verifier import create_citation_mapping_from_sources

sources = [
    {'index': '1', 'url': 'https://example.com', 'content': '...'},
    {'index': '2', 'url': 'https://en.wikipedia.org/wiki/DMD', 'content': '...'}
]

# Wikipedia sources are automatically excluded
citation_map = create_citation_mapping_from_sources(sources)
# citation_map only contains non-Wikipedia sources
```

## Important Notes

1. **Reference vs. Citation**: Wikipedia is used as a **reference article** for comparison (ROUGE scores, entity recall), but Wikipedia URLs are **banned** from being used as **citation sources** in the generated article.

2. **Coded Implementation**: The ban is implemented in code, not just specified in natural language. The `is_wikipedia_url()` function performs actual URL parsing and domain checking.

3. **Automatic Filtering**: Citation mapping and verification automatically exclude Wikipedia sources without requiring manual intervention.

4. **Warning Messages**: The system prints warnings when Wikipedia sources are detected, making it clear that they are being excluded.

## Testing

To verify the Wikipedia ban is working:

1. Run evaluation on a generated article that contains Wikipedia URLs
2. Check console output for warnings about Wikipedia sources
3. Verify that Wikipedia URLs are excluded from citation verification
4. Confirm that only non-Wikipedia sources are used in citation metrics

## Files Modified

- `eval/eval_article_content.py`: Added Wikipedia detection and filtering functions
- `eval/citation_verifier.py`: Added Wikipedia filtering to citation mapping
- `eval/WIKIPEDIA_BAN_IMPLEMENTATION.md`: This documentation file

