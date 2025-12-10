"""
Article Content Evaluation Module

Implements evaluation metrics for rare disease articles' content:
1. ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) - Lin, 2004
2. Entity recall using FLAIR NER - article-level entity recall

To assess the full-length article quality, we adopt ROUGE scores (Lin, 2004) 
and compute the entity recall at the article level based on FLAIR NER results.
"""

from rouge_score import rouge_scorer
from transformers import pipeline
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# Wikipedia Domain Ban - Coded Implementation
# ============================================================================

WIKIPEDIA_DOMAINS = {
    'wikipedia.org',
    'wikipedia.com',
    'wikimedia.org',
    'wikidata.org',
    'wiktionary.org',
    'wikiquote.org',
    'wikibooks.org',
    'wikisource.org',
    'wikinews.org',
    'wikiversity.org',
    'wikivoyage.org',
    'mediawiki.org',
    'foundation.wikimedia.org'
}

WIKIPEDIA_SUBDOMAINS = {
    'en.wikipedia.org',
    'es.wikipedia.org',
    'fr.wikipedia.org',
    'de.wikipedia.org',
    'it.wikipedia.org',
    'pt.wikipedia.org',
    'ru.wikipedia.org',
    'ja.wikipedia.org',
    'zh.wikipedia.org',
    'ar.wikipedia.org',
    # Add other language subdomains as needed
}


def is_wikipedia_url(url: str) -> bool:
    """
    Check if a URL belongs to Wikipedia or any Wikimedia project.
    This is a coded check, not just natural language specification.
    
    Args:
        url: URL string to check
    
    Returns:
        True if URL is from Wikipedia/Wikimedia, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        parsed = urlparse(url.lower().strip())
        domain = parsed.netloc.lower()
        
        # Check exact domain match
        if domain in WIKIPEDIA_DOMAINS:
            return True
        
        # Check if domain ends with any Wikipedia domain
        for wiki_domain in WIKIPEDIA_DOMAINS:
            if domain.endswith('.' + wiki_domain) or domain == wiki_domain:
                return True
        
        # Check subdomain matches
        if domain in WIKIPEDIA_SUBDOMAINS:
            return True
        
        # Check for 'wikipedia' in domain (catch variations)
        if 'wikipedia' in domain or 'wikimedia' in domain:
            return True
        
        return False
    except Exception:
        # If parsing fails, check if 'wikipedia' or 'wiki' is in the URL string
        url_lower = url.lower()
        return 'wikipedia' in url_lower or ('wiki' in url_lower and 'media' in url_lower)


def filter_wikipedia_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter out Wikipedia URLs from a list of URLs.
    
    Args:
        urls: List of URL strings
    
    Returns:
        Tuple of (filtered_urls, wikipedia_urls_removed)
    """
    filtered = []
    removed = []
    
    for url in urls:
        if is_wikipedia_url(url):
            removed.append(url)
        else:
            filtered.append(url)
    
    return filtered, removed


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract all URLs from text using regex.
    
    Args:
        text: Text to extract URLs from
    
    Returns:
        List of URL strings found in text
    """
    # Pattern to match URLs (http, https, www, etc.)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    return urls


def check_text_for_wikipedia_sources(text: str) -> Dict[str, any]:
    """
    Check text for Wikipedia sources and return statistics.
    
    Args:
        text: Text to check
    
    Returns:
        Dictionary with statistics about Wikipedia sources found
    """
    urls = extract_urls_from_text(text)
    wikipedia_urls = [url for url in urls if is_wikipedia_url(url)]
    non_wikipedia_urls = [url for url in urls if not is_wikipedia_url(url)]
    
    return {
        'total_urls': len(urls),
        'wikipedia_urls': wikipedia_urls,
        'wikipedia_url_count': len(wikipedia_urls),
        'non_wikipedia_urls': non_wikipedia_urls,
        'non_wikipedia_url_count': len(non_wikipedia_urls),
        'has_wikipedia_sources': len(wikipedia_urls) > 0
    }


def calculate_rouge_scores(generated_article: str, reference_article: str) -> Dict[str, float]:
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.
    
    Args:
        generated_article: The generated article text
        reference_article: The reference article text
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_article, generated_article)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def extract_entities(text: str, ner_pipeline) -> set:
    """
    Extract named entities from text using Hugging Face transformers NER.
    
    Args:
        text: Input text
        ner_pipeline: Hugging Face NER pipeline
    
    Returns:
        Set of unique entity strings
    """
    # Split into sentences for processing
    sentences = re.split(r'[.!?]\s+', text)
    entities = set()
    
    for sent in sentences:
        if len(sent.strip()) < 3:
            continue
        try:
            results = ner_pipeline(sent)
            for result in results:
                entity_text = result.get('word', result.get('entity_group', ''))
                if entity_text and len(entity_text.strip()) > 1:
                    entities.add(entity_text.strip())
        except Exception:
            continue  # Skip if processing fails
    
    return entities


def calculate_entity_recall(generated_article: str, reference_article: str, tagger) -> float:
    """
    Calculate entity recall: proportion of reference entities found in generated article.
    
    Args:
        generated_article: The generated article text
        reference_article: The reference article text
        tagger: FLAIR SequenceTagger model
    
    Returns:
        Entity recall score (0.0 to 1.0)
    """
    reference_entities = extract_entities(reference_article, tagger)
    generated_entities = extract_entities(generated_article, tagger)
    
    if len(reference_entities) == 0:
        return 1.0 if len(generated_entities) == 0 else 0.0
    
    # Calculate recall: entities in both sets / total reference entities
    matched_entities = reference_entities.intersection(generated_entities)
    recall = len(matched_entities) / len(reference_entities)
    
    return recall


def extract_citations(text: str) -> List[str]:
    """
    Extract citation markers from text (e.g., [1], [2], [1][2]).
    Note: This function only extracts citation numbers, not URLs.
    Wikipedia URL filtering is handled separately in citation verification.
    
    Args:
        text: Article text with citations
    
    Returns:
        List of unique citation numbers as strings
    """
    # Pattern matches [n] or [n][m] etc.
    citation_pattern = r'\[(\d+)\]'
    citations = re.findall(citation_pattern, text)
    return list(set(citations))




def fetch_wikipedia_article(url: str) -> Optional[str]:
    """
    Fetch and extract main content from a Wikipedia article.
    Uses Wikipedia's REST API to get plain text content.
    
    Args:
        url: Wikipedia article URL
    
    Returns:
        Cleaned article text content, or None if fetch fails
    """
    try:
        # Convert URL to API format for plain text
        # e.g., https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy
        # -> https://en.wikipedia.org/api/rest_v1/page/summary/Duchenne_muscular_dystrophy
        # For full text, use: https://en.wikipedia.org/api/rest_v1/page/html/...
        page_name = url.split('/wiki/')[-1]
        
        # Try to get plain text version first (simpler)
        # If that doesn't work, fall back to HTML parsing
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{page_name}"
        
        response = requests.get(
            api_url, 
            headers={'User-Agent': 'Article-Evaluator/1.0 (Educational Research)'},
            timeout=10
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove navigation, references, and other non-content elements
        for element in soup.find_all(['nav', 'aside', 'style', 'script', 'link']):
            element.decompose()
        
        # Remove citation references (they're in <sup> tags with class 'reference')
        for sup in soup.find_all('sup', class_='reference'):
            sup.decompose()
        
        # Remove infoboxes and tables (they're not part of main content for evaluation)
        for table in soup.find_all('table'):
            # Keep some tables if they're informative, but remove navigation boxes
            if 'infobox' in table.get('class', []):
                table.decompose()
        
        # Extract text from main content
        # Wikipedia HTML structure: main content is in <body>
        main_content = soup.find('body')
        if main_content:
            # Get all paragraph and heading text
            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            text_parts = []
            for p in paragraphs:
                text = p.get_text(separator=' ', strip=True)
                # Filter out very short fragments and navigation text
                if text and len(text) > 20 and not text.startswith('Jump to'):
                    text_parts.append(text)
            
            return '\n\n'.join(text_parts)
        
        return None
        
    except Exception as e:
        print(f"Error fetching Wikipedia article: {e}")
        return None


def clean_wikipedia_text(text: str) -> str:
    """
    Clean Wikipedia text by removing common Wikipedia artifacts.
    
    Args:
        text: Raw Wikipedia text
    
    Returns:
        Cleaned text
    """
    # Remove edit links and other Wikipedia-specific markers
    text = re.sub(r'\[edit\]', '', text)
    text = re.sub(r'Jump to.*?hide', '', text, flags=re.DOTALL)
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def visualize_rouge_scores(rouge_scores: Dict[str, float], output_path: str = "rouge_scores.png"):
    """Create bar chart of ROUGE scores."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    scores = [rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL']]
    
    bars = ax.bar(metrics, scores, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('ROUGE Scores: Generated vs Wikipedia Reference', fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 ROUGE visualization saved: {output_path}")


def visualize_entity_recall(
    entity_recall: float,
    reference_count: int,
    generated_count: int,
    matched_count: int,
    output_path: str = "entity_recall.png"
):
    """Create visualization of entity recall metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Recall score
    ax1.barh(['Entity Recall'], [entity_recall], color='#06A77D', alpha=0.8, height=0.5)
    ax1.set_xlim([0, 1.0])
    ax1.set_xlabel('Recall Score', fontweight='bold')
    ax1.set_title('Entity Recall Score', fontweight='bold')
    ax1.text(entity_recall + 0.05, 0, f'{entity_recall:.3f}', va='center', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Right: Entity counts
    categories = ['Reference', 'Generated', 'Matched']
    counts = [reference_count, generated_count, matched_count]
    bars = ax2.bar(categories, counts, color=['#2E86AB', '#A23B72', '#06A77D'], alpha=0.8)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Number of Entities', fontweight='bold')
    ax2.set_title('Entity Counts', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('FLAIR NER Entity Recall Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Entity recall visualization saved: {output_path}")


if __name__ == "__main__":
    # Reference article: Wikipedia article on Duchenne muscular dystrophy
    # NOTE: We fetch Wikipedia for reference comparison, but we BAN Wikipedia
    # sources from being used as citations in the generated article
    reference_url = "https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy"
    
    # Load generated article
    generated_article_path = "/Users/nnataliewang19/Documents/coterm q/fall cs 224v/conv-rare-disease/src/output/report_fabry_disease_20251130_191249.md"
    try:
        with open(generated_article_path, 'r', encoding='utf-8') as f:
            generated_article = f.read()
        print(f"Loaded generated article from {generated_article_path}")
        
        # Check for Wikipedia sources in generated article (BANNED)
        print("\n" + "="*60)
        print("CHECKING FOR BANNED WIKIPEDIA SOURCES IN GENERATED ARTICLE")
        print("="*60)
        wiki_check = check_text_for_wikipedia_sources(generated_article)
        
        if wiki_check['has_wikipedia_sources']:
            print(f"WARNING: Found {wiki_check['wikipedia_url_count']} Wikipedia URL(s) in generated article!")
            print("   Wikipedia sources are BANNED and should be removed:")
            for url in wiki_check['wikipedia_urls']:
                print(f"     - {url}")
            print("\n   These URLs will be excluded from citation verification.")
        else:
            print("No Wikipedia sources found in generated article.")
        
        print(f"\n   Total URLs found: {wiki_check['total_urls']}")
        print(f"   Non-Wikipedia URLs: {wiki_check['non_wikipedia_url_count']}")
        print("="*60 + "\n")
        
    except FileNotFoundError:
        print(f"Error: Could not find {generated_article_path}")
        generated_article = ""
    
    # Fetch reference article from Wikipedia (for comparison only)
    print(f"Fetching reference article from {reference_url}...")
    print("(Note: Wikipedia is used as REFERENCE for comparison, not as a citation source)")
    reference_article_raw = fetch_wikipedia_article(reference_url)
    
    if reference_article_raw:
        reference_article = clean_wikipedia_text(reference_article_raw)
        print(f"Successfully fetched reference article ({len(reference_article)} characters)")
        
        # Calculate ROUGE scores
        print("\nCalculating ROUGE scores...")
        rouge_scores = calculate_rouge_scores(generated_article, reference_article)
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        # Visualize ROUGE scores
        output_dir = Path("eval_outputs")
        output_dir.mkdir(exist_ok=True)
        visualize_rouge_scores(rouge_scores, str(output_dir / "rouge_scores.png"))
        
        # Calculate entity recall using Hugging Face Hub models (no local download)
        print("\nCalculating entity recall (loading NER model from Hugging Face Hub)...")
        try:
            # Load model directly from Hugging Face Hub - cached but not permanently stored
            # Using a high-quality NER model similar to FLAIR's capabilities
            ner_pipeline = pipeline(
                "ner", 
                model="dslim/bert-base-NER",  # High-quality NER model from Hugging Face
                aggregation_strategy="simple",
                device=-1  # Use CPU (set to 0 for GPU if available)
            )
            entity_recall = calculate_entity_recall(generated_article, reference_article, ner_pipeline)
            print(f"Entity recall: {entity_recall:.4f}")
            
            # Show entity statistics
            reference_entities = extract_entities(reference_article, ner_pipeline)
            generated_entities = extract_entities(generated_article, ner_pipeline)
            matched_entities = reference_entities.intersection(generated_entities)
            
            print(f"\nEntity statistics:")
            print(f"  Reference entities: {len(reference_entities)}")
            print(f"  Generated entities: {len(generated_entities)}")
            print(f"  Matched entities: {len(matched_entities)}")
            
            # Visualize entity recall
            visualize_entity_recall(
                entity_recall,
                len(reference_entities),
                len(generated_entities),
                len(matched_entities),
                str(output_dir / "entity_recall.png")
            )
            
        except Exception as e:
            print(f"Error loading NER model: {e}")
        
        # Note: Wikipedia criteria evaluation is handled by vertex_evaluator.py
        # which uses Vertex AI Gemini models (more powerful than Prometheus)
        print("\nNote: Wikipedia criteria evaluation (Interest, Coherence, Relevance, Coverage, Verifiability)")
        print("      is handled separately using Vertex AI Gemini models via vertex_evaluator.py")
        
    else:
        print("Error: Could not fetch reference article from Wikipedia")
