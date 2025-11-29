"""
Vertex AI (Gemini) based Wikipedia Criteria Evaluation

Uses Vertex AI Gemini models to evaluate articles on 5 Wikipedia criteria:
1. Interest Level
2. Coherence and Organization
3. Relevance and Focus
4. Coverage
5. Verifiability

Based on Wikipedia criteria, we evaluate the article from the aspects of:
(1) Interest Level, (2) Coherence and Organization, (3) Relevance and Focus, 
(4) Coverage, and (5) Verifiability.

Uses Gemini 2.5 Pro for high-quality evaluation.
"""

import os
import re
from typing import Dict, Any, Optional
from google.cloud import aiplatform
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def create_vertex_client(project_id: Optional[str] = None, location: str = "us-central1", gcp_api_key: Optional[str] = None):
    """
    Create a Vertex AI client for Gemini models.
    
    Args:
        project_id: GCP project ID
        location: GCP location (default: us-central1)
        gcp_api_key: Path to service account JSON file (optional)
    
    Returns:
        VertexLLMClient instance
    """
    try:
        from src.wiki_storm_pipeline import VertexLLMClient, create_vertex_client as create_client
        return create_client(project_id=project_id, location=location, gcp_api_key=gcp_api_key)
    except ImportError:
        # Fallback: create client directly
        if gcp_api_key and os.path.exists(gcp_api_key):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_api_key
        
        if not project_id:
            project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        if not project_id:
            raise ValueError("GCP_PROJECT_ID environment variable not set")
        
        aiplatform.init(project=project_id, location=location)
        
        class SimpleVertexClient:
            def __init__(self, project_id, location):
                self.project_id = project_id
                self.location = location
            
            def generate(self, prompt: str, *, model: str, temperature: float, max_tokens: int, **kwargs) -> str:
                try:
                    from vertexai.preview.generative_models import GenerativeModel
                except ImportError:
                    from vertexai.generative_models import GenerativeModel
                
                config = {"temperature": temperature, "max_output_tokens": max_tokens}
                if kwargs.get("reasoning_effort"):
                    config["reasoning_effort"] = kwargs["reasoning_effort"]
                
                model_obj = GenerativeModel(model)
                response = model_obj.generate_content(prompt, generation_config=config)
                return response.text
        
        return SimpleVertexClient(project_id, location)


def get_wikipedia_rubric(aspect: str) -> str:
    """
    Get the 1-5 scale rubric for a Wikipedia evaluation aspect.
    
    Args:
        aspect: One of 'interest', 'coherence', 'relevance', 'coverage', 'verifiability'
    
    Returns:
        Rubric string for Gemini evaluation
    """
    rubrics = {
        'interest': """Score 1: Not engaging at all; no attempt to capture the reader’s attention.
Score 2: Fairly engaging with a basic narrative but lacking depth.
Score 3: Moderately engaging with several interesting points.
Score 4: Quite engaging with a well-structured narrative and noteworthy points that frequently capture and retain attention.
Score 5: Exceptionally engaging throughout, with a compelling narrative that consistently stimulates interest.""",
        
        'coherence': """Score 1: Disorganized; lacks logical structure and coherence.
Score 2: Fairly organized; a basic structure is present but not consistently followed.
Score 3: Organized; a clear structure is mostly followed with some lapses in coherence.
Score 4: Good organization; a clear structure with minor lapses in coherence.
Score 5: Excellently organized; the article is logically structured with seamless transitions and a clear argument.""",
        
        'relevance': """Score 1: Off-topic; the content does not align with the headline or core subject.
Score 2: Somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to.
Score 3: Generally on topic, despite a few unrelated details.
Score 4: Mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions.
Score 5: Exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing to a comprehensive understanding of the topic.""",
        
        'coverage': """Score 1: Severely lacking; offers little to no coverage of the topic's primary aspects, resulting in a very narrow perspective.
Score 2: Partial coverage; includes some of the topic's main aspects but misses others, resulting in an incomplete portrayal.
Score 3: Acceptable breadth; covers most main aspects, though it may stray into minor unnecessary details or overlook some relevant points.
Score 4: Good coverage; achieves broad coverage of the topic, hitting on all major points with minimal extraneous information.
Score 5: Exemplary in breadth; delivers outstanding coverage, thoroughly detailing all crucial aspects of the topic without including irrelevant information.""",
        
        'verifiability': """Score 1: No citations or sources; claims cannot be verified.
Score 2: Very few citations; most claims lack support.
Score 3: Some citations present but many important claims lack support.
Score 4: Well-cited with most claims supported by reliable sources.
Score 5: Excellently cited with all claims supported by high-quality, verifiable sources."""
    }
    
    return rubrics.get(aspect.lower(), "")


def trim_article_to_word_limit(text: str, max_words: int = 3000) -> str:
    """
    Trim article to max_words by truncating from the end if needed.
    Gemini 2.5 Pro can handle longer context, so we use 3000 words.
    
    Args:
        text: Article text
        max_words: Maximum number of words
    
    Returns:
        Trimmed article text
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    
    return ' '.join(words[:max_words])


def evaluate_with_vertex_ai(
    article: str, 
    aspect: str, 
    vertex_client, 
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Evaluate article using Vertex AI Gemini model on a specific aspect.
    
    Args:
        article: Article text (will be trimmed to 3000 words if needed)
        aspect: Evaluation aspect ('interest', 'coherence', 'relevance', 'coverage')
        vertex_client: Vertex AI client instance
        model_name: Gemini model name (default: gemini-2.5-pro)
        temperature: Generation temperature (default: 0.2 for consistent evaluation)
    
    Returns:
        Dictionary with score and feedback
    """
    trimmed_article = trim_article_to_word_limit(article, max_words=3000)
    
    rubric = get_wikipedia_rubric(aspect)
    
    # Build evaluation prompt for Gemini - use simple, direct format
    prompt = f"""Evaluate this article on {aspect.capitalize()} for a rare disease article (patients/families audience).

Article:
{trimmed_article}

Rubric:
{rubric}

Provide your evaluation in this exact format:

Feedback: [your assessment]

Score: [1, 2, 3, 4, or 5]

Evaluation:"""
    
    try:
        # Call Vertex AI - increase max_tokens to ensure we get the score
        response = vertex_client.generate(
            prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=2048  # Increased to ensure score is included
        )
        
        if not response:
            return {
                'score': None,
                'feedback': "Empty response from Vertex AI",
                'aspect': aspect,
                'error': 'Empty response'
            }
        
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response)
        
        # Normalize response: remove markdown formatting, normalize whitespace
        response_normalized = re.sub(r'\*\*|__|\*|_', '', response)  # Remove bold/italic
        response_normalized = re.sub(r'\s+', ' ', response_normalized)  # Normalize whitespace
        
        # Debug: print first 300 chars of response
        print(f"    Response preview: {response[:300]}...")
        
        # Extract score - try multiple approaches
        score = None
        
        # Approach 1: Try explicit patterns (case-insensitive, flexible whitespace)
        score_patterns = [
            r'Score\s*:?\s*(\d+)',
            r'score\s*:?\s*(\d+)',
            r'\[RESULT\]\s*(\d+)',
            r'(\d+)\s*/?\s*5',
            r'score\s+of\s+(\d+)',
            r'rating\s*:?\s*(\d+)',
            r'grade\s*:?\s*(\d+)'
        ]
        
        for pattern in score_patterns:
            score_match = re.search(pattern, response_normalized, re.IGNORECASE)
            if score_match:
                try:
                    score = int(score_match.group(1))
                    if 1 <= score <= 5:
                        break
                except (ValueError, IndexError):
                    continue
        
        # Approach 2: Find any number 1-5 near "score" keyword
        if not score:
            # Look in a wider context around "score"
            score_context = re.search(r'(?:score|rating|grade).{0,50}(\d+)', response_normalized, re.IGNORECASE)
            if score_context:
                try:
                    num = int(score_context.group(1))
                    if 1 <= num <= 5:
                        score = num
                except (ValueError, IndexError):
                    pass
        
        # Approach 3: Look for score after "Feedback:" section (common pattern)
        if not score:
            # Split by "Feedback:" and look for score in the second part
            parts = re.split(r'Feedback\s*:', response_normalized, flags=re.IGNORECASE)
            if len(parts) > 1:
                after_feedback = parts[1]
                # Look for "Score:" or just a number 1-5 in this section
                score_match = re.search(r'Score\s*:?\s*(\d+)', after_feedback, re.IGNORECASE)
                if score_match:
                    try:
                        score = int(score_match.group(1))
                        if 1 <= score <= 5:
                            pass  # score is set
                    except (ValueError, IndexError):
                        pass
        
        # Approach 4: Find any standalone digit 1-5 (prefer numbers near end)
        if not score:
            all_numbers = re.findall(r'\b([1-5])\b', response_normalized)
            if all_numbers:
                # Prefer numbers in the last 100 chars (where score usually appears)
                last_part = response_normalized[-100:]
                last_numbers = re.findall(r'\b([1-5])\b', last_part)
                if last_numbers:
                    score = int(last_numbers[-1])
                else:
                    score = int(all_numbers[-1])
        
        # Approach 5: If response ends without score, try to infer from feedback quality
        # Look for quality indicators in the last part of response
        if not score:
            last_part = response_normalized[-200:].lower()
            # Look for quality words that might indicate score
            if any(word in last_part for word in ['excellent', 'outstanding', 'exceptional', 'exemplary']):
                score = 5
            elif any(word in last_part for word in ['very good', 'quite good', 'well', 'strong']):
                score = 4
            elif any(word in last_part for word in ['adequate', 'acceptable', 'moderate', 'reasonable']):
                score = 3
            elif any(word in last_part for word in ['poor', 'lacking', 'weak', 'inadequate']):
                score = 2
            elif any(word in last_part for word in ['very poor', 'severely', 'extremely poor']):
                score = 1
        
        # Approach 6: Absolute last resort - find ANY digit 1-5 in response
        if not score:
            # Search the original response (not normalized) for any 1-5
            for char in reversed(response):
                if char.isdigit() and '1' <= char <= '5':
                    score = int(char)
                    break
        
        # Extract feedback
        feedback = response.strip()
        if score:
            # If we found a score, try to extract feedback before it
            feedback_match = re.search(r'(.+?)(?:Score:|score:|\d+/5|$)', response, re.DOTALL | re.IGNORECASE)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
                if len(feedback) < 20:  # If too short, use full response
                    feedback = response.strip()
            print(f"    ✅ Extracted score: {score}")
        else:
            print(f"    ⚠️  Could not extract score from response")
            print(f"    Full response length: {len(response)} chars")
            print(f"    Last 200 chars: {response[-200:]}")
        
        return {
            'score': score,
            'feedback': feedback,
            'aspect': aspect,
            'model': model_name,
            'raw_response': response[:500] if not score else None  # Include raw response if score extraction failed
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"    ⚠️  Error: {str(e)}")
        return {
            'score': None,
            'feedback': f"Error during evaluation: {str(e)}",
            'aspect': aspect,
            'error': str(e),
            'traceback': error_details
        }


def evaluate_all_aspects_vertex(
    article: str,
    vertex_client=None,
    model_name: str = "gemini-2.5-pro",
    project_id: Optional[str] = None,
    location: str = "us-central1",
    gcp_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate article on all 5 Wikipedia criteria aspects using Vertex AI Gemini.
    
    Based on Wikipedia criteria, evaluates: (1) Interest Level, (2) Coherence and Organization,
    (3) Relevance and Focus, (4) Coverage, and (5) Verifiability.
    
    Args:
        article: Article text to evaluate
        vertex_client: Vertex AI client (will be created if not provided)
        model_name: Gemini model name (default: gemini-2.5-pro)
        project_id: GCP project ID (required if client not provided)
        location: GCP location (default: us-central1)
        gcp_api_key: Path to service account JSON (optional)
    
    Returns:
        Dictionary with scores for all 5 aspects
    """
    if vertex_client is None:
        if not project_id:
            project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        vertex_client = create_vertex_client(project_id=project_id, location=location, gcp_api_key=gcp_api_key)
    
    aspects = ['interest', 'coherence', 'relevance', 'coverage', 'verifiability']
    results = {}
    
    print(f"Evaluating article with {model_name} on {len(aspects)} Wikipedia criteria aspects...")
    
    for aspect in aspects:
        print(f"  Evaluating {aspect}...")
        result = evaluate_with_vertex_ai(article, aspect, vertex_client, model_name)
        results[aspect] = result
    
    # Visualize results
    output_dir = Path("eval_outputs")
    output_dir.mkdir(exist_ok=True)
    visualize_wikipedia_criteria(results, model_name, str(output_dir / "wikipedia_criteria.png"))
    
    return results


def visualize_wikipedia_criteria(results: Dict[str, Any], model_name: str = "gemini-2.5-pro", output_path: str = "wikipedia_criteria.png"):
    """Create bar chart of Wikipedia criteria scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    aspects = ['Interest', 'Coherence', 'Relevance', 'Coverage', 'Verifiability']
    scores = []
    labels = []
    for aspect in ['interest', 'coherence', 'relevance', 'coverage', 'verifiability']:
        score = results.get(aspect, {}).get('score')
        if score:
            scores.append(score)
            labels.append(f'{score}/5')
        else:
            scores.append(0)
            labels.append('N/A')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62828']
    bars = ax.bar(aspects, scores, color=colors, alpha=0.8)
    
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=11)
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.2,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=11, color='red')
    
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Wikipedia Criteria Evaluation ({model_name})', fontweight='bold', fontsize=14)
    ax.set_ylim([0, 5.5])
    ax.set_yticks(range(0, 6))
    ax.grid(axis='y', alpha=0.3)
    
    # Add average score line
    valid_scores = [s for s in scores if s]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        ax.axhline(y=avg_score, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg_score:.2f}')
        ax.legend()
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Wikipedia criteria visualization saved: {output_path}")


if __name__ == "__main__":
    # Load article to evaluate
    article_path = Path("/Users/nnataliewang19/Documents/coterm q/fall cs 224v/conv-rare-disease/src/outputs/20251128-134246_wiki_min/hierarchical_report_RESULT.md")
    
    print("="*80)
    print("Wikipedia Criteria Evaluation (Vertex AI Gemini)")
    print("="*80)
    print(f"\n📄 Loading article from: {article_path}")
    
    try:
        with open(article_path, 'r', encoding='utf-8') as f:
            article = f.read()
        print(f"✅ Loaded article ({len(article)} characters)")
    except FileNotFoundError:
        print(f"❌ Error: Article not found at {article_path}")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading article: {e}")
        exit(1)
    
    # Check GCP configuration
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    gcp_api_key = os.environ.get("GCP_API_KEY")
    
    if not project_id:
        print("\n❌ Error: GCP_PROJECT_ID not set")
        print("   Set it with: export GCP_PROJECT_ID=your-project-id")
        exit(1)
    
    print(f"\n🔧 Using GCP Project: {project_id}")
    print(f"   Model: gemini-2.5-pro")
    print()
    
    # Run evaluation
    try:
        results = evaluate_all_aspects_vertex(
            article,
            model_name="gemini-2.5-pro",
            project_id=project_id,
            gcp_api_key=gcp_api_key
        )
        
        # Print results
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS")
        print("="*80)
        
        scores = {}
        for aspect, result in results.items():
            score = result.get('score')
            if score:
                scores[aspect] = score
                print(f"\n{aspect.capitalize()}: {score}/5")
                if result.get('feedback'):
                    feedback = result['feedback'][:200].replace('\n', ' ')
                    print(f"  {feedback}...")
            else:
                print(f"\n{aspect.capitalize()}: Error - {result.get('error', 'No score returned')}")
                if result.get('raw_response'):
                    print(f"  Raw response: {result['raw_response']}")
                if result.get('traceback'):
                    print(f"  Traceback: {result['traceback'][:300]}...")
        
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            print(f"\n{'='*80}")
            print(f"Average Score: {avg_score:.2f}/5")
            print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

