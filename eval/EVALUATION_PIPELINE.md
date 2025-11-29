# Evaluation Pipeline Documentation

## Overview

The evaluation pipeline compares a **generated article** against the **actual Wikipedia article** to assess quality. The pipeline enforces a **coded ban on Wikipedia domains** - Wikipedia sources cannot be used as citations in the generated article.

## Pipeline Structure

```
Generated Article → Evaluation → Wikipedia Reference Article
                              ↓
                    [Wikipedia Domain Ban Enforced]
                              ↓
                    Multiple Evaluation Metrics
```

## Evaluation Metrics

### 1. ROUGE Scores
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence
- Compares generated article with Wikipedia reference

### 2. Entity Recall
- Extracts named entities from both articles using NER
- Calculates: matched entities / reference entities
- Uses BERT-based NER model

### 3. Wikipedia Criteria Evaluation (Vertex AI Gemini 2.5 Pro)
Based on Wikipedia criteria, evaluates articles from 5 aspects:
- **Interest Level**: How engaging and accessible the article is
- **Coherence and Organization**: Structure and flow
- **Relevance and Focus**: How well it stays on topic
- **Coverage**: Comprehensiveness of information
- **Verifiability**: Quality and completeness of citations (Wikipedia sources banned)
- Uses **Gemini 2.5 Pro** for high-quality evaluation

### 4. Citation Verification
- Citation recall: Proportion of cited sentences that are entailed by sources
- Citation precision: Proportion of citations that entail their sentences
- **Wikipedia sources are automatically excluded** (coded ban)

## Wikipedia Domain Ban

The ban is **coded**, not just specified in natural language:

- `is_wikipedia_url()` function checks URLs against banned domains
- Citation mapping automatically filters out Wikipedia sources
- Evaluation reports warnings when Wikipedia URLs are detected
- Wikipedia is used as **reference** for comparison, not as a **citation source**

## Usage

### Command Line

```bash
# Basic evaluation
python eval/run_evaluation.py \
    --generated src/outputs/20251128-134246_wiki_min/hierarchical_report_RESULT.md \
    --wikipedia-url https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy \
    --output evaluation_results.json

# With citation sources
python eval/run_evaluation.py \
    --generated generated_article.md \
    --wikipedia-url https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy \
    --citation-sources citation_sources.json \
    --output evaluation_results.json \
    --model gemini-2.5-pro
```

### Python API

```python
from eval.run_evaluation import run_full_evaluation

results = run_full_evaluation(
    generated_article_path="generated_article.md",
    wikipedia_url="https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy",
    citation_sources=[...],  # Optional
    output_file="results.json",  # Optional
    vertex_model="gemini-2.5-pro"
)

print(f"ROUGE-L: {results['rouge_scores']['rougeL']:.4f}")
print(f"Entity Recall: {results['entity_recall']['recall']:.4f}")
print(f"Average Criteria Score: {results['wikipedia_criteria']['average_score']:.2f}/5")
```

## Available Vertex AI Models

The evaluation uses Vertex AI Gemini models (more powerful than Prometheus):

- **gemini-2.5-pro** (recommended): Most powerful, best for evaluation
- **gemini-2.5-flash**: Faster, still high quality
- **gemini-2.0-flash-001**: Good balance of speed and quality
- **gemini-1.5-pro-002**: Previous generation, still excellent

Default: `gemini-2.5-pro`

## Environment Setup

```bash
# Required environment variables
export GCP_PROJECT_ID=your-project-id
export GCP_API_KEY=/path/to/service-account.json  # Optional if using gcloud auth

# Or use gcloud authentication
gcloud auth application-default login
export GCP_PROJECT_ID=your-project-id
```

## Output Format

The evaluation returns a JSON dictionary with:

```json
{
  "generated_article_path": "...",
  "wikipedia_url": "...",
  "wikipedia_source_check": {
    "has_wikipedia_sources": false,
    "wikipedia_url_count": 0,
    ...
  },
  "rouge_scores": {
    "rouge1": 0.45,
    "rouge2": 0.32,
    "rougeL": 0.42
  },
  "entity_recall": {
    "recall": 0.78,
    "reference_entities_count": 150,
    "generated_entities_count": 120,
    "matched_entities_count": 117
  },
  "wikipedia_criteria": {
    "interest": {"score": 4, "feedback": "..."},
    "coherence": {"score": 4, "feedback": "..."},
    "relevance": {"score": 5, "feedback": "..."},
    "coverage": {"score": 4, "feedback": "..."},
    "average_score": 4.25
  },
  "citation_metrics": {
    "citation_recall": 0.85,
    "citation_precision": 0.92,
    ...
  }
}
```

## Key Features

1. ✅ **Wikipedia Domain Ban**: Coded implementation, not just documentation
2. ✅ **Powerful Models**: Uses Gemini 2.5 Pro instead of Prometheus
3. ✅ **Comprehensive Metrics**: ROUGE, entity recall, criteria, citations
4. ✅ **Clear Output**: Detailed console output and JSON results
5. ✅ **Error Handling**: Graceful handling of missing components

## Example: Evaluating Generated Report

```bash
# Evaluate the generated hierarchical report
python eval/run_evaluation.py \
    --generated src/outputs/20251128-134246_wiki_min/hierarchical_report_RESULT.md \
    --wikipedia-url https://en.wikipedia.org/wiki/Duchenne_muscular_dystrophy \
    --output eval_results.json \
    --model gemini-2.5-pro
```

This will:
1. Load the generated article
2. Check for banned Wikipedia sources
3. Fetch the Wikipedia reference article
4. Calculate all metrics
5. Save results to JSON

## Notes

- Wikipedia is used as a **reference** for comparison (ROUGE, entity recall)
- Wikipedia sources are **banned** from citations in the generated article
- The ban is enforced in code, not just specified in prompts
- Vertex AI models are more powerful than Prometheus and provide better evaluation

