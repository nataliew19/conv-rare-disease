"""
Main Evaluation Pipeline

Evaluates generated article against Wikipedia reference article with:
1. ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
2. Entity recall using NER
3. Wikipedia criteria evaluation using Vertex AI Gemini 2.5 Pro (4 aspects)
4. Citation verification (with Wikipedia domain ban enforced)

The evaluation compares the generated article with the actual Wikipedia article,
ensuring Wikipedia sources are BANNED from citations.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_article_content import (
    calculate_rouge_scores,
    calculate_entity_recall,
    extract_entities,
    fetch_wikipedia_article,
    clean_wikipedia_text,
    check_text_for_wikipedia_sources,
    extract_citations
)
from vertex_evaluator import evaluate_all_aspects_vertex
from citation_verifier import (
    calculate_citation_metrics,
    create_citation_mapping_from_sources,
    load_mistral_model
)


def run_full_evaluation(
    generated_article_path: str,
    wikipedia_url: str,
    citation_sources: Optional[list] = None,
    output_file: Optional[str] = None,
    use_vertex_ai: bool = True,
    vertex_model: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    Args:
        generated_article_path: Path to generated article file
        wikipedia_url: URL of Wikipedia article to use as reference
        citation_sources: List of source documents for citation verification (optional)
        output_file: Path to save evaluation results JSON (optional)
        use_vertex_ai: Use Vertex AI Gemini instead of Prometheus (default: True)
        vertex_model: Gemini model name (default: gemini-2.5-pro)
    
    Returns:
        Dictionary with all evaluation results
    """
    print("="*80)
    print("EVALUATION PIPELINE: Generated Article vs Wikipedia Reference")
    print("="*80)
    print()
    
    # Load generated article
    print(f"📄 Loading generated article from: {generated_article_path}")
    try:
        with open(generated_article_path, 'r', encoding='utf-8') as f:
            generated_article = f.read()
        print(f"   ✅ Loaded ({len(generated_article)} characters)")
    except FileNotFoundError:
        print(f"   ❌ Error: File not found")
        return {'error': f'Generated article not found: {generated_article_path}'}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {'error': str(e)}
    
    # Check for Wikipedia sources (BANNED)
    print("\n" + "="*80)
    print("🔍 CHECKING FOR BANNED WIKIPEDIA SOURCES")
    print("="*80)
    wiki_check = check_text_for_wikipedia_sources(generated_article)
    
    if wiki_check['has_wikipedia_sources']:
        print(f"⚠️  WARNING: Found {wiki_check['wikipedia_url_count']} Wikipedia URL(s) in generated article!")
        print("   Wikipedia sources are BANNED and will be excluded from citation verification:")
        for url in wiki_check['wikipedia_urls']:
            print(f"     - {url}")
    else:
        print("✅ No Wikipedia sources found in generated article.")
    
    print(f"\n   Total URLs: {wiki_check['total_urls']}")
    print(f"   Non-Wikipedia URLs: {wiki_check['non_wikipedia_url_count']}")
    print("="*80)
    
    # Fetch Wikipedia reference article
    print(f"\n📚 Fetching Wikipedia reference article: {wikipedia_url}")
    print("   (Note: Wikipedia is used as REFERENCE for comparison, not as a citation source)")
    reference_article_raw = fetch_wikipedia_article(wikipedia_url)
    
    if not reference_article_raw:
        print("   ❌ Error: Could not fetch Wikipedia article")
        return {'error': 'Could not fetch Wikipedia reference article'}
    
    reference_article = clean_wikipedia_text(reference_article_raw)
    print(f"   ✅ Fetched ({len(reference_article)} characters)")
    
    results = {
        'generated_article_path': generated_article_path,
        'wikipedia_url': wikipedia_url,
        'wikipedia_source_check': wiki_check
    }
    
    # 1. ROUGE Scores
    print("\n" + "="*80)
    print("1️⃣  CALCULATING ROUGE SCORES")
    print("="*80)
    rouge_scores = calculate_rouge_scores(generated_article, reference_article)
    results['rouge_scores'] = rouge_scores
    print(f"   ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"   ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"   ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # 2. Entity Recall
    print("\n" + "="*80)
    print("2️⃣  CALCULATING ENTITY RECALL")
    print("="*80)
    try:
        from transformers import pipeline
        print("   Loading NER model...")
        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=-1
        )
        
        entity_recall = calculate_entity_recall(generated_article, reference_article, ner_pipeline)
        reference_entities = extract_entities(reference_article, ner_pipeline)
        generated_entities = extract_entities(generated_article, ner_pipeline)
        matched_entities = reference_entities.intersection(generated_entities)
        
        results['entity_recall'] = {
            'recall': entity_recall,
            'reference_entities_count': len(reference_entities),
            'generated_entities_count': len(generated_entities),
            'matched_entities_count': len(matched_entities)
        }
        
        print(f"   Entity Recall: {entity_recall:.4f}")
        print(f"   Reference entities: {len(reference_entities)}")
        print(f"   Generated entities: {len(generated_entities)}")
        print(f"   Matched entities: {len(matched_entities)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['entity_recall'] = {'error': str(e)}
    
    # 3. Wikipedia Criteria Evaluation (Vertex AI Gemini - 5 aspects)
    print("\n" + "="*80)
    print("3️⃣  EVALUATING WIKIPEDIA CRITERIA (Vertex AI Gemini)")
    print("   Aspects: Interest, Coherence, Relevance, Coverage, Verifiability")
    print("="*80)
    
    if use_vertex_ai:
        try:
            project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            gcp_api_key = os.environ.get("GCP_API_KEY")
            
            if not project_id:
                print("   ⚠️  Warning: GCP_PROJECT_ID not set, skipping Vertex AI evaluation")
                results['wikipedia_criteria'] = {'error': 'GCP_PROJECT_ID not set'}
            else:
                print(f"   Using {vertex_model} for evaluation...")
                criteria_results = evaluate_all_aspects_vertex(
                    generated_article,
                    model_name=vertex_model,
                    project_id=project_id,
                    gcp_api_key=gcp_api_key
                )
                
                results['wikipedia_criteria'] = criteria_results
                
                print(f"\n   📊 Wikipedia Criteria Scores:")
                scores = {}
                for aspect, result in criteria_results.items():
                    score = result.get('score')
                    if score:
                        scores[aspect] = score
                        print(f"      {aspect.capitalize()}: {score}/5")
                        if result.get('feedback'):
                            feedback_preview = result['feedback'][:100].replace('\n', ' ')
                            print(f"         {feedback_preview}...")
                
                if scores:
                    avg_score = sum(scores.values()) / len(scores)
                    print(f"\n      Average: {avg_score:.2f}/5")
                    results['wikipedia_criteria']['average_score'] = avg_score
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results['wikipedia_criteria'] = {'error': str(e)}
    else:
        print("   Skipping (use_vertex_ai=False)")
        results['wikipedia_criteria'] = {'skipped': True}
    
    # 4. Citation Verification
    print("\n" + "="*80)
    print("4️⃣  CITATION VERIFICATION (Wikipedia Sources BANNED)")
    print("="*80)
    
    if citation_sources:
        try:
            print("   Creating citation mapping (Wikipedia sources will be excluded)...")
            citation_map = create_citation_mapping_from_sources(citation_sources)
            
            print("   Loading Mistral model for citation verification...")
            mistral_model, mistral_tokenizer = load_mistral_model()
            
            print("   Calculating citation metrics...")
            citation_metrics = calculate_citation_metrics(
                generated_article,
                citation_map,
                model=mistral_model,
                tokenizer=mistral_tokenizer,
                filter_wikipedia=True
            )
            
            results['citation_metrics'] = citation_metrics
            
            print(f"\n   📊 Citation Metrics:")
            print(f"      Citation Recall: {citation_metrics['citation_recall']:.4f}")
            print(f"      Citation Precision: {citation_metrics['citation_precision']:.4f}")
            print(f"      Total sentences: {citation_metrics['total_sentences']}")
            print(f"      Cited sentences: {citation_metrics['cited_sentences']}")
            print(f"      Entailed sentences: {citation_metrics['entailed_sentences']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results['citation_metrics'] = {'error': str(e)}
    else:
        print("   ⚠️  No citation sources provided, skipping citation verification")
        results['citation_metrics'] = {'skipped': True, 'reason': 'No citation sources provided'}
    
    # Summary
    print("\n" + "="*80)
    print("📋 EVALUATION SUMMARY")
    print("="*80)
    
    if 'rouge_scores' in results:
        print(f"   ROUGE-L: {results['rouge_scores']['rougeL']:.4f}")
    
    if 'entity_recall' in results and 'recall' in results['entity_recall']:
        print(f"   Entity Recall: {results['entity_recall']['recall']:.4f}")
    
    if 'wikipedia_criteria' in results and 'average_score' in results['wikipedia_criteria']:
        print(f"   Wikipedia Criteria (avg): {results['wikipedia_criteria']['average_score']:.2f}/5")
    
    if 'citation_metrics' in results and 'citation_recall' in results['citation_metrics']:
        print(f"   Citation Recall: {results['citation_metrics']['citation_recall']:.4f}")
    
    print("="*80)
    
    # Save results
    if output_file:
        print(f"\n💾 Saving results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("   ✅ Saved")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate generated article against Wikipedia reference')
    parser.add_argument('--generated', type=str, required=True,
                        help='Path to generated article file')
    parser.add_argument('--wikipedia-url', type=str, required=True,
                        help='Wikipedia article URL to use as reference')
    parser.add_argument('--citation-sources', type=str,
                        help='Path to JSON file with citation sources (optional)')
    parser.add_argument('--output', type=str,
                        help='Path to save evaluation results JSON (optional)')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro',
                        help='Vertex AI Gemini model name (default: gemini-2.5-pro)')
    
    args = parser.parse_args()
    
    # Load citation sources if provided
    citation_sources = None
    if args.citation_sources:
        try:
            with open(args.citation_sources, 'r', encoding='utf-8') as f:
                citation_sources = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load citation sources: {e}")
    
    results = run_full_evaluation(
        generated_article_path=args.generated,
        wikipedia_url=args.wikipedia_url,
        citation_sources=citation_sources,
        output_file=args.output,
        vertex_model=args.model
    )

