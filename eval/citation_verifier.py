"""
Citation Verification using Mistral 7B-Instruct

Uses Mistral 7B-Instruct to verify whether cited passages entail the generated sentences.
Calculates citation recall and precision based on entailment checking.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch
from typing import Dict, List, Tuple, Optional, Any
from urllib.parse import urlparse

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
        
        # Check for 'wikipedia' in domain (catch variations)
        if 'wikipedia' in domain or 'wikimedia' in domain:
            return True
        
        return False
    except Exception:
        # If parsing fails, check if 'wikipedia' or 'wiki' is in the URL string
        url_lower = url.lower()
        return 'wikipedia' in url_lower or ('wiki' in url_lower and 'media' in url_lower)


def extract_sentences_with_citations(text: str) -> List[Dict[str, Any]]:
    """
    Extract sentences from article along with their citation markers.
    
    Args:
        text: Article text with citations
    
    Returns:
        List of dictionaries with 'sentence' and 'citations' fields
    """
    # Split by sentence endings (period, exclamation, question mark)
    sentences = re.split(r'([.!?]+\s+)', text)
    
    # Recombine sentences with their punctuation
    sentence_list = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        
        # Extract citations from this sentence
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, sentence)
        
        # Remove citations from sentence text for cleaner comparison
        sentence_clean = re.sub(r'\[\d+\]', '', sentence).strip()
        
        if sentence_clean and len(sentence_clean) > 10:  # Filter very short fragments
            sentence_list.append({
                'sentence': sentence_clean,
                'citations': list(set(citations)),  # Unique citation numbers
                'original': sentence.strip()
            })
    
    return sentence_list


def check_entailment(premise: str, hypothesis: str, model, tokenizer) -> Dict[str, Any]:
    """
    Check if premise entails hypothesis using Mistral 7B-Instruct.
    
    Args:
        premise: The cited passage (premise)
        hypothesis: The generated sentence (hypothesis)
        model: Mistral model
        tokenizer: Mistral tokenizer
    
    Returns:
        Dictionary with entailment result and confidence
    """
    # Create prompt for entailment checking
    prompt = f"""<s>[INST] Determine if the premise entails (supports) the hypothesis. Respond with only "ENTAILS" or "NOT_ENTAILS".

Premise: {premise}

Hypothesis: {hypothesis}

Response: [/INST]"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to device
    try:
        if hasattr(model, 'device'):
            device = model.device
        elif hasattr(model, 'hf_device_map'):
            device = next(iter(model.hf_device_map.values()))
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,  # Use greedy decoding for deterministic results
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response (everything after [/INST])
    if '[/INST]' in response:
        response = response.split('[/INST]')[-1].strip()
    
    # Check if entails
    entails = "ENTAILS" in response.upper()
    
    return {
        'entails': entails,
        'response': response,
        'premise': premise,
        'hypothesis': hypothesis
    }


def verify_citation_entailment(
    sentence: str, 
    citation_num: str, 
    cited_passage: str, 
    model, 
    tokenizer
) -> bool:
    """
    Verify if a cited passage entails a sentence.
    
    Args:
        sentence: Generated sentence
        citation_num: Citation number
        cited_passage: The passage that was cited
    
    Returns:
        True if passage entails sentence, False otherwise
    """
    result = check_entailment(cited_passage, sentence, model, tokenizer)
    return result['entails']


def calculate_citation_metrics(
    article: str,
    citation_to_passage: Dict[str, str],
    model=None,
    tokenizer=None,
    filter_wikipedia: bool = True
) -> Dict[str, Any]:
    """
    Calculate citation recall and precision.
    Wikipedia sources are BANNED and excluded from verification.
    
    Citation Recall: Proportion of sentences with citations that are entailed by their cited passages
    Citation Precision: Proportion of cited passages that entail their sentences
    
    Args:
        article: Generated article text
        citation_to_passage: Dictionary mapping citation numbers to source passages
        model: Mistral model (will be loaded if not provided)
        tokenizer: Mistral tokenizer (will be loaded if not provided)
        filter_wikipedia: If True, filter out Wikipedia sources (default: True)
    
    Returns:
        Dictionary with recall, precision, and detailed results
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_mistral_model()
    
    # Note: Wikipedia filtering should be done when creating citation_to_passage
    # using create_citation_mapping_from_sources() which filters Wikipedia URLs
    
    # Extract sentences with citations
    sentences_with_citations = extract_sentences_with_citations(article)
    
    # Filter to only sentences that have citations
    cited_sentences = [s for s in sentences_with_citations if s['citations']]
    
    if not cited_sentences:
        return {
            'citation_recall': 0.0,
            'citation_precision': 0.0,
            'total_sentences': len(sentences_with_citations),
            'cited_sentences': 0,
            'entailed_sentences': 0,
            'total_citations': 0,
            'entailed_citations': 0
        }
    
    # Check entailment for each sentence-citation pair
    entailed_sentences = 0
    entailed_citations = 0
    total_citations = 0
    detailed_results = []
    
    for sent_data in cited_sentences:
        sentence = sent_data['sentence']
        citations = sent_data['citations']
        
        sentence_entailed = False
        citation_results = []
        
        for citation_num in citations:
            total_citations += 1
            
            # Get the cited passage
            cited_passage = citation_to_passage.get(citation_num, "")
            
            if not cited_passage:
                # Citation not found in mapping - count as not entailed
                citation_results.append({
                    'citation': citation_num,
                    'entails': False,
                    'reason': 'Citation not found in source mapping'
                })
                continue
            
            # Check entailment
            entails = verify_citation_entailment(sentence, citation_num, cited_passage, model, tokenizer)
            
            if entails:
                entailed_citations += 1
                sentence_entailed = True
            
            citation_results.append({
                'citation': citation_num,
                'entails': entails,
                'passage': cited_passage[:100] + "..." if len(cited_passage) > 100 else cited_passage
            })
        
        if sentence_entailed:
            entailed_sentences += 1
        
        detailed_results.append({
            'sentence': sentence[:100] + "..." if len(sentence) > 100 else sentence,
            'citations': citations,
            'entailed': sentence_entailed,
            'citation_results': citation_results
        })
    
    # Calculate metrics
    citation_recall = entailed_sentences / len(cited_sentences) if cited_sentences else 0.0
    citation_precision = entailed_citations / total_citations if total_citations > 0 else 0.0
    
    return {
        'citation_recall': citation_recall,
        'citation_precision': citation_precision,
        'total_sentences': len(sentences_with_citations),
        'cited_sentences': len(cited_sentences),
        'entailed_sentences': entailed_sentences,
        'total_citations': total_citations,
        'entailed_citations': entailed_citations,
        'detailed_results': detailed_results
    }


def load_mistral_model(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Load Mistral 7B-Instruct model with quantization if available.
    
    Args:
        model_name: Hugging Face model name
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Try 8-bit loading first, fall back to regular loading
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config
        )
    except (ImportError, Exception):
        # Fall back to regular loading with float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    return model, tokenizer


def create_citation_mapping_from_sources(sources: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Create a mapping from citation numbers to source passages.
    Wikipedia sources are BANNED and will be excluded (coded check, not just natural language).
    
    Args:
        sources: List of source documents with 'index'/'id', 'url', 'content'/'text' fields
    
    Returns:
        Dictionary mapping citation numbers (as strings) to passage text
        (Wikipedia sources are excluded)
    """
    citation_map = {}
    wikipedia_sources_removed = []
    
    for source in sources:
        # Get citation number
        citation_num = str(source.get('index', source.get('id', '')))
        if not citation_num:
            continue
        
        # Check if source has a URL and if it's from Wikipedia (BANNED)
        url = source.get('url', '')
        if url:
            # Check URL string directly
            url_str = url if isinstance(url, str) else str(url)
            if is_wikipedia_url(url_str):
                wikipedia_sources_removed.append({
                    'citation': citation_num,
                    'url': url_str
                })
                continue  # Skip Wikipedia sources - BANNED
        
        # Also check if URL is embedded in content/text
        content = source.get('content', source.get('text', ''))
        if content and isinstance(content, str):
            # Extract URLs from content and check if any are Wikipedia
            import re
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls_in_content = re.findall(url_pattern, content)
            if any(is_wikipedia_url(url) for url in urls_in_content):
                wikipedia_sources_removed.append({
                    'citation': citation_num,
                    'url': urls_in_content[0] if urls_in_content else 'embedded in content'
                })
                continue  # Skip Wikipedia sources - BANNED
        
        # Add non-Wikipedia source to mapping
        if content:
            citation_map[citation_num] = content
    
    if wikipedia_sources_removed:
        print(f"\n⚠️  WARNING: Excluded {len(wikipedia_sources_removed)} Wikipedia source(s) from citation mapping (BANNED):")
        for removed in wikipedia_sources_removed:
            print(f"     Citation [{removed['citation']}]: {removed['url']}")
        print()
    
    return citation_map


# ============================================================================
# Command-line execution
# ============================================================================

def load_article(article_path: str) -> str:
    """Load article text from file."""
    with open(article_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_rag_response(rag_response_path: str):
    """Load RAG response from JSON file."""
    import json
    import sys
    from pathlib import Path
    
    # Try to import RagResponse - handle import path issues
    try:
        from src.src.dataclass import RagResponse
    except ImportError:
        try:
            # Try alternative import path
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from src.src.dataclass import RagResponse
        except ImportError:
            raise ImportError("Could not import RagResponse. Make sure src/src/dataclass.py is accessible.")
    
    with open(rag_response_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return RagResponse.from_dict(data)


def convert_rag_documents_to_sources(rag_response) -> list[dict]:
    """
    Convert RAG response documents to format expected by citation_verifier.
    
    Args:
        rag_response: RagResponse object with cited_documents
    
    Returns:
        List of source dictionaries with 'index', 'url', 'content' fields
    """
    sources = []
    for i, doc in enumerate(rag_response.cited_documents, start=1):
        # Combine all excerpts into a single content string
        content = " ".join(doc.excerpts) if doc.excerpts else ""
        
        sources.append({
            'index': str(i),  # Citation number (1-indexed)
            'id': str(i),
            'url': doc.url,
            'content': content,
            'text': content,  # Also provide as 'text' for compatibility
            'title': doc.title
        })
    
    return sources


def main():
    """Main entry point for command-line execution."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify citations in an article using Mistral 7B-Instruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval/citation_verifier.py article.md rag_response.json
  python eval/citation_verifier.py src/output/report.md src/output/rag_response.json
        """
    )
    parser.add_argument(
        'article_path',
        type=str,
        help='Path to article file (markdown with citations like [1], [2], etc.)'
    )
    parser.add_argument(
        'rag_response_path',
        type=str,
        help='Path to RAG response JSON file (from pipeline_wrapper.py output)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help='Hugging Face model name (default: mistralai/Mistral-7B-Instruct-v0.2)'
    )
    parser.add_argument(
        '--no-filter-wikipedia',
        action='store_true',
        help='Do not filter out Wikipedia sources (default: Wikipedia sources are excluded)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Citation Verification")
    print("="*80)
    print(f"\n📄 Loading article from: {args.article_path}")
    print(f"📄 Loading RAG response from: {args.rag_response_path}")
    
    # Load article
    try:
        article = load_article(args.article_path)
        print(f"✅ Loaded article ({len(article)} characters)")
    except FileNotFoundError:
        print(f"❌ Error: Article not found at {args.article_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading article: {e}")
        sys.exit(1)
    
    # Load RAG response
    try:
        rag_response = load_rag_response(args.rag_response_path)
        print(f"✅ Loaded RAG response ({len(rag_response.cited_documents)} cited documents)")
    except FileNotFoundError:
        print(f"❌ Error: RAG response not found at {args.rag_response_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading RAG response: {e}")
        sys.exit(1)
    
    # Convert RAG documents to citation mapping format
    print("\n🔄 Converting sources to citation mapping...")
    sources = convert_rag_documents_to_sources(rag_response)
    citation_mapping = create_citation_mapping_from_sources(sources)
    print(f"✅ Created citation mapping ({len(citation_mapping)} citations)")
    
    # Load Mistral model (this may take a while on first run)
    print("\n🤖 Loading Mistral 7B-Instruct model...")
    print("   (This may take a few minutes on first run)")
    try:
        model, tokenizer = load_mistral_model(model_name=args.model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure you have transformers and torch installed:")
        print("   pip install transformers torch")
        sys.exit(1)
    
    # Calculate citation metrics
    print("\n🔍 Verifying citations...")
    print("   (This may take a while depending on article length)")
    
    try:
        results = calculate_citation_metrics(
            article=article,
            citation_to_passage=citation_mapping,
            model=model,
            tokenizer=tokenizer,
            filter_wikipedia=not args.no_filter_wikipedia
        )
        
        # Print results
        print("\n" + "="*80)
        print("📊 CITATION VERIFICATION RESULTS")
        print("="*80)
        print(f"\nCitation Recall: {results['citation_recall']:.2%}")
        print(f"  ({results['entailed_sentences']}/{results['cited_sentences']} cited sentences are entailed)")
        print(f"\nCitation Precision: {results['citation_precision']:.2%}")
        print(f"  ({results['entailed_citations']}/{results['total_citations']} citations are entailed)")
        print(f"\nTotal Sentences: {results['total_sentences']}")
        print(f"Cited Sentences: {results['cited_sentences']}")
        print(f"Total Citations: {results['total_citations']}")
        
        # Show some detailed results
        if results.get('detailed_results'):
            print("\n" + "-"*80)
            print("📝 Sample Detailed Results (first 5):")
            print("-"*80)
            for i, detail in enumerate(results['detailed_results'][:5], 1):
                print(f"\n{i}. Sentence: {detail['sentence']}")
                print(f"   Citations: {detail['citations']}")
                print(f"   Entailed: {'✅ Yes' if detail['entailed'] else '❌ No'}")
                for cit_result in detail['citation_results']:
                    status = "✅" if cit_result['entails'] else "❌"
                    reason = cit_result.get('reason', '')
                    print(f"     [{cit_result['citation']}] {status} {reason}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n❌ Error during citation verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()





