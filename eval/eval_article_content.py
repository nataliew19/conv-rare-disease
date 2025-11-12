"""
Article Content Evaluation Module

Implements multiple evaluation metrics for rare disease articles' content:
1. ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
2. Entity recall using FLAIR NER
3. Wikipedia criteria evaluation using Prometheus (4 aspects)
4. Citation recall and precision using Mistral 7B-Instruct
"""

from rouge_score import rouge_scorer
from transformers import pipeline
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Optional
from prometheus_evaluator import load_prometheus_model, evaluate_all_aspects


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


if __name__ == "__main__":
    # Reference article: Wikipedia article on Duchenne muscular dystrophy
    reference_url = "https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy"
    
    # Load generated article
    generated_article_path = "../generated_example_DMD.md"
    try:
        with open(generated_article_path, 'r', encoding='utf-8') as f:
            generated_article = f.read()
        print(f"Loaded generated article from {generated_article_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {generated_article_path}")
        generated_article = ""
    
    # Fetch reference article from Wikipedia
    print(f"Fetching reference article from {reference_url}...")
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
            
        except Exception as e:
            print(f"Error loading NER model: {e}")
        
        # Evaluate with Prometheus (4 Wikipedia criteria)
        print("\nEvaluating with Prometheus (Wikipedia criteria)...")
        try:
            print("Loading Prometheus model (this may take a while on first use)...")
            model, tokenizer = load_prometheus_model()
            print("Prometheus model loaded successfully.")
            
            results = evaluate_all_aspects(generated_article, model, tokenizer)
            
            print(f"\nPrometheus Scores Summary:")
            scores = {}
            for aspect, result in results.items():
                score = result['score']
                scores[aspect] = score
                print(f"  {aspect.capitalize()}: {score}/5")
                if result['feedback']:
                    print(f"    Feedback: {result['feedback'][:150]}...")
            
            if scores:
                print(f"  Average: {sum(scores.values()) / len(scores):.2f}/5")
            
        except Exception as e:
            print(f"Error loading Prometheus model: {e}")
            print("Prometheus model will be downloaded from Hugging Face Hub on first use.")
            print("Note: This is a large model (~13B parameters) and requires significant memory.")
        
    else:
        print("Error: Could not fetch reference article from Wikipedia")
