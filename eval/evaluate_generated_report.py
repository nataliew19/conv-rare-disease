#!/usr/bin/env python3
"""
Quick evaluation script for the generated hierarchical report.

This script evaluates the generated report against the Wikipedia article
for Duchenne Muscular Dystrophy.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_evaluation import run_full_evaluation

# Configuration
GENERATED_ARTICLE = Path(__file__).parent.parent / "src" / "outputs" / "20251128-134246_wiki_min" / "hierarchical_report_RESULT.md"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy"
OUTPUT_FILE = Path(__file__).parent / "evaluation_results.json"
VERTEX_MODEL = "gemini-2.5-pro"  # Most powerful model

if __name__ == "__main__":
    print("🚀 Starting Evaluation Pipeline")
    print(f"   Generated article: {GENERATED_ARTICLE}")
    print(f"   Wikipedia reference: {WIKIPEDIA_URL}")
    print(f"   Vertex AI model: {VERTEX_MODEL}")
    print()
    
    # Check if generated article exists
    if not GENERATED_ARTICLE.exists():
        print(f"❌ Error: Generated article not found at {GENERATED_ARTICLE}")
        print("   Please update the path in this script.")
        sys.exit(1)
    
    # Check GCP configuration
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("⚠️  Warning: GCP_PROJECT_ID not set. Wikipedia criteria evaluation will be skipped.")
        print("   Set it with: export GCP_PROJECT_ID=your-project-id")
        print()
    
    # Run evaluation
    results = run_full_evaluation(
        generated_article_path=str(GENERATED_ARTICLE),
        wikipedia_url=WIKIPEDIA_URL,
        citation_sources=None,  # Add citation sources if available
        output_file=str(OUTPUT_FILE),
        use_vertex_ai=True,
        vertex_model=VERTEX_MODEL
    )
    
    if 'error' in results:
        print(f"\n❌ Evaluation failed: {results['error']}")
        sys.exit(1)
    
    print(f"\n✅ Evaluation complete! Results saved to: {OUTPUT_FILE}")
    print("\n📊 Quick Summary:")
    
    if 'rouge_scores' in results:
        print(f"   ROUGE-L: {results['rouge_scores']['rougeL']:.4f}")
    
    if 'entity_recall' in results and 'recall' in results['entity_recall']:
        print(f"   Entity Recall: {results['entity_recall']['recall']:.4f}")
    
    if 'wikipedia_criteria' in results and 'average_score' in results['wikipedia_criteria']:
        print(f"   Wikipedia Criteria (avg): {results['wikipedia_criteria']['average_score']:.2f}/5")

