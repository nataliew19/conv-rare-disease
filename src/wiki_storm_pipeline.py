"""STORM-style pipeline for rare-disease Wikipedia guideline research."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from guideline import Guideline, generate_guideline

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: simple .env loader
    def load_dotenv():
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_path):
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    load_dotenv()

try:
    from google.cloud import aiplatform
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SectionsScope:
    required_sections: Optional[List[str]] = None
    optional_sections: List[str] = field(default_factory=list)
    omit_sections: List[str] = field(default_factory=list)


@dataclass
class CitationPreferences:
    priority_sources: List[str] = field(default_factory=list)
    forbidden_sources: List[str] = field(default_factory=list)
    format: str = "numeric"


@dataclass
class GuidelineInput:
    rare_disease: str
    article_style: str = "patient_facing"
    audience_profile: Optional[str] = None
    sections_scope: Optional[SectionsScope] = None
    writing_heuristics: List[str] = field(default_factory=list)
    citation_preferences: Optional[CitationPreferences] = None
    style_notes: Optional[str] = None


@dataclass
class LLMConfig:
    """LLM configuration.
    
    Available Gemini models: gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash-001,
    gemini-1.5-pro-002, gemini-1.5-flash-002, text-bison@002
    
    Note: Gemini models support max_output_tokens up to 8192.
    """
    model_name: str = "gemini-2.0-flash-001"  # Newer available model
    temperature: float = 0.4
    max_tokens: int = 8192  # Maximum supported by Gemini models
    reasoning_effort: Optional[str] = None


class LLMClient(Protocol):
    def generate(self, prompt: str, *, model: str, temperature: float, max_tokens: int, **kwargs: Any) -> str: ...


# ============================================================================
# Vertex AI Client
# ============================================================================

class VertexLLMClient:
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1", gcp_api_key: Optional[str] = None):
        if not VERTEX_AVAILABLE:
            raise ImportError("google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform")
        
        # If service account file provided (and it's actually a file path), use it
        if gcp_api_key and os.path.exists(gcp_api_key):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_api_key
            # Try to extract project_id from JSON if not provided
            if not project_id:
                try:
                    with open(gcp_api_key, 'r') as f:
                        project_id = json.load(f).get("project_id")
                except (json.JSONDecodeError, IOError):
                    pass
        # If gcp_api_key is set but not a file, ignore it (might be an API key string)
        # We'll use Application Default Credentials instead
        
        # Get project_id from environment if still not set
        if not project_id:
            project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        if not project_id:
            raise ValueError(
                "project_id is required. Set GCP_PROJECT_ID env var, provide it directly, "
                "or ensure service account JSON contains 'project_id'"
            )
        
        # Initialize Vertex AI (will use ADC if GOOGLE_APPLICATION_CREDENTIALS not set)
        aiplatform.init(project=project_id, location=location)
        self.project_id = project_id

    def generate(self, prompt: str, *, model: str, temperature: float, max_tokens: int, **kwargs: Any) -> str:
        is_gemini = "gemini" in model.lower()
        
        if is_gemini:
            try:
                from vertexai.preview.generative_models import GenerativeModel
            except ImportError:
                from vertexai.generative_models import GenerativeModel
            
            # Normalize Gemini model names for Vertex AI
            # Try different naming formats that Vertex AI might use
            model_variants = [model]  # Try original first
            
            # Support newer Gemini 2.5 and 2.0 models
            if "gemini-2.5-pro" in model.lower():
                model_variants = ["gemini-2.5-pro", "gemini-2.0-flash-001", "gemini-1.5-pro-002"]
            elif "gemini-2.5-flash" in model.lower():
                model_variants = ["gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-1.5-flash-002"]
            elif "gemini-2.0-flash" in model.lower():
                model_variants = ["gemini-2.0-flash-001", "gemini-2.5-flash", "gemini-1.5-flash-002"]
            elif "gemini-1.5-pro" in model.lower():
                model_variants = ["gemini-1.5-pro-002", "gemini-1.5-pro", "gemini-2.0-flash-001", "gemini-pro"]
            elif "gemini-1.5-flash" in model.lower():
                model_variants = ["gemini-1.5-flash-002", "gemini-1.5-flash", "gemini-2.0-flash-001", "gemini-flash"]
            elif "gemini-pro" in model.lower() and "2" not in model.lower():
                model_variants = ["gemini-pro", "gemini-2.0-flash-001", "gemini-1.5-pro-002"]
            
            config = {"temperature": temperature, "max_output_tokens": max_tokens}
            if kwargs.get("reasoning_effort"):
                config["reasoning_effort"] = kwargs["reasoning_effort"]
            
            # Try each model variant until one works
            last_error = None
            for variant in model_variants:
                try:
                    return GenerativeModel(variant).generate_content(prompt, generation_config=config).text
                except Exception as e:
                    last_error = e
                    if "404" not in str(e) and "not found" not in str(e).lower():
                        # Not a model not found error, re-raise
                        raise
                    continue
            
            # If all variants failed
            raise RuntimeError(
                f"None of the Gemini model variants {model_variants} are available. "
                f"Last error: {last_error}. "
                f"Try using 'text-bison@002' instead, or check available models in GCP Console."
            ) from last_error
        else:
            model_obj = aiplatform.TextGenerationModel.from_pretrained(model)
            params = {"temperature": temperature, "max_output_tokens": max_tokens}
            if kwargs.get("reasoning_effort"):
                params["reasoning_effort"] = kwargs["reasoning_effort"]
            
            response = model_obj.predict(prompt, **params)
            return response.text if hasattr(response, "text") else str(response)


def create_vertex_client(project_id: Optional[str] = None, location: str = "us-central1", gcp_api_key: Optional[str] = None) -> VertexLLMClient:
    return VertexLLMClient(project_id=project_id, location=location, gcp_api_key=gcp_api_key)


# ============================================================================
# Guideline Building
# ============================================================================

def build_custom_guideline(config: GuidelineInput, llm_client: Optional[LLMClient] = None, llm_config: Optional[LLMConfig] = None) -> Guideline:
    """Build guideline with Wikipedia article discovery and extraction."""
    from guideline import find_wikipedia_articles, extract_wikipedia_info
    
    wikipedia_info = None
    if llm_client and llm_config:
        articles = find_wikipedia_articles(config.rare_disease, llm_client, llm_config)
        if articles:
            wikipedia_info = extract_wikipedia_info(articles, config.rare_disease, llm_client, llm_config)
    
    base = generate_guideline(config.rare_disease, config.article_style, wikipedia_info=wikipedia_info)
    
    sections = _merge_sections(base.sections, config.sections_scope)
    heuristics = base.heuristics + config.writing_heuristics
    audience = config.audience_profile or base.audience
    
    citation_resources = base.citation_resources
    citation_rules = list(base.citation_rules)
    
    if config.citation_preferences:
        if config.citation_preferences.priority_sources:
            citation_resources = _prioritize_sources(config.citation_preferences.priority_sources, citation_resources)
        if config.citation_preferences.forbidden_sources:
            citation_rules.append(f"Avoid citing: {', '.join(config.citation_preferences.forbidden_sources)}")
        if config.citation_preferences.format:
            citation_rules.insert(0, f"Use {config.citation_preferences.format} citation format.")
    
    style_notes = base.style_notes
    if config.style_notes:
        style_notes = f"{style_notes}\n\nCustom notes:\n{config.style_notes}".strip()
    
    return replace(base, audience=audience, sections=sections, heuristics=heuristics,
                   citation_resources=citation_resources, citation_rules=citation_rules, style_notes=style_notes)


def _merge_sections(base_sections: List[str], scope: Optional[SectionsScope]) -> List[str]:
    if not scope:
        return list(base_sections)
    
    sections = list(scope.required_sections) if scope.required_sections else list(base_sections)
    sections.extend(s for s in scope.optional_sections if s not in sections)
    
    if scope.omit_sections:
        sections = [s for s in sections if s not in scope.omit_sections]
    
    return sections


def _prioritize_sources(priority: List[str], base_sources: List[str]) -> List[str]:
    ordered = []
    for source in priority:
        if source and source not in ordered:
            ordered.append(source)
    ordered.extend(s for s in base_sources if s not in ordered)
    return ordered


# ============================================================================
# STORM Plan Generation
# ============================================================================

STORM_PERSPECTIVES = ["Clinical Pharmacologist", "Patient Advocate", "Rare-disease Historian", "Molecular Geneticist"]


def generate_storm_plan(guideline: Guideline, llm: LLMClient, config: LLMConfig) -> Dict[str, Any]:
    prompt = f"""You are STORM, a multi-perspective Wikipedia researcher.
Create a JSON plan with fields:
  "meta": summary + prioritized questions,
  "nodes": list of {{
      "perspective": <one of {STORM_PERSPECTIVES}>,
      "objective": <short goal>,
      "tasks": [<steps>],
      "suggested_sources": [<resource or URL keywords, including Wikipedia sections to review>],
      "handoff": <how this feeds the next persona>
  }} in execution order.

Include Wikipedia sections and key resources in suggested_sources.

Guideline to respect:
{guideline}
"""
    kwargs = {"reasoning_effort": config.reasoning_effort} if config.reasoning_effort else {}
    response = llm.generate(prompt.strip(), model=config.model_name, temperature=config.temperature,
                           max_tokens=config.max_tokens, **kwargs)
    return _parse_json(response)


def synthesize_hierarchical_report(plan: Dict[str, Any], llm: LLMClient, config: LLMConfig) -> str:
    prompt = f"""You are compiling a hierarchical research report for a rare-disease Wikipedia article.

Use this plan JSON:
{json.dumps(plan, indent=2)}

Output Markdown with:
- Executive Summary (bullet list of key findings, cite strongest evidence)
- Research Tree (H1/H2/H3 headings mirroring plan order)
- Evidence Table mapping sections to sources
- Next Questions / Gaps
"""
    return llm.generate(prompt.strip(), model=config.model_name, temperature=0.2,
                       max_tokens=config.max_tokens).strip()


def _parse_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) > 1:
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nPreview: {cleaned[:200]}...") from e


# ============================================================================
# Main Pipeline
# ============================================================================

def run_wiki_storm_pipeline(config: GuidelineInput, llm_client: LLMClient, llm_config: LLMConfig) -> Dict[str, Any]:
    """Run end-to-end pipeline: Wikipedia discovery → guideline → plan → report."""
    print("Finding Wikipedia articles and building guidelines...")
    guideline = build_custom_guideline(config, llm_client, llm_config)
    print("Generating STORM plan...")
    storm_plan = generate_storm_plan(guideline, llm_client, llm_config)
    return {
        "guideline": guideline.to_dict(),
        "plan": storm_plan,
        "report_markdown": synthesize_hierarchical_report(storm_plan, llm_client, llm_config),
    }


__all__ = [
    "SectionsScope", "CitationPreferences", "GuidelineInput", "LLMConfig", "LLMClient",
    "VertexLLMClient", "create_vertex_client", "build_custom_guideline",
    "generate_storm_plan", "synthesize_hierarchical_report", "run_wiki_storm_pipeline",
]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys

    # Try multiple possible env var names (service account file is optional)
    gcp_api_key = (os.environ.get("GCP_API_KEY") or 
                   os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or
                   os.environ.get("GCP_CREDENTIALS"))
    project_id = (os.environ.get("GCP_PROJECT_ID") or 
                 os.environ.get("GOOGLE_CLOUD_PROJECT"))
    
    # Debug: show what we found
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        print("Environment variables:")
        print(f"  GCP_API_KEY: {os.environ.get('GCP_API_KEY', 'NOT SET')}")
        print(f"  GCP_PROJECT_ID: {os.environ.get('GCP_PROJECT_ID', 'NOT SET')}")
        print(f"  GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'NOT SET')}")
        print(f"  Resolved gcp_api_key: {gcp_api_key or 'Using Application Default Credentials'}")
        print(f"  Resolved project_id: {project_id}")
        print()
    
    if not project_id:
        print("ERROR: GCP_PROJECT_ID environment variable not set")
        print("\nYou have two authentication options:")
        print("\nOption 1 - Use gcloud auth (NO service account needed):")
        print("  1. Run: gcloud auth application-default login")
        print("  2. Set: export GCP_PROJECT_ID=your-project-id")
        print("  3. Run the script")
        print("\nOption 2 - Use service account JSON file:")
        print("  1. Create service account in GCP Console")
        print("  2. Download JSON key file")
        print("  3. Set: export GCP_API_KEY=/path/to/service-account.json")
        print("  4. Set: export GCP_PROJECT_ID=your-project-id")
        sys.exit(1)
    
    # Only use gcp_api_key if it's actually a file path
    service_account_file = gcp_api_key if (gcp_api_key and os.path.exists(gcp_api_key)) else None
    
    if not service_account_file:
        print("INFO: Using Application Default Credentials (no service account file needed)")
        print("      Make sure you've run: gcloud auth application-default login")
    
    try:
        config = GuidelineInput(rare_disease="Duchenne Muscular Dystrophy", article_style="patient_facing")
        llm_config = LLMConfig(model_name="gemini-2.0-flash-001", temperature=0.4, max_tokens=8192)
        llm_client = create_vertex_client(project_id=project_id, gcp_api_key=service_account_file)
        
        print("Running pipeline...")
        results = run_wiki_storm_pipeline(config, llm_client, llm_config)
        
        print(f"\n✓ Guideline: {len(results['guideline']['sections'])} sections")
        print(f"✓ Plan: {len(results['plan'].get('nodes', []))} nodes")
        print(f"✓ Report: {len(results['report_markdown'])} chars")
        
        # Save results to files
        disease_safe = config.rare_disease.replace(" ", "_").replace("/", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_file = f"storm_report_{disease_safe}_{timestamp}.md"
        plan_file = f"storm_plan_{disease_safe}_{timestamp}.json"
        guideline_file = f"storm_guideline_{disease_safe}_{timestamp}.json"
        
        # Save report markdown
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(results['report_markdown'])
        print(f"\n✓ Report saved to: {report_file}")
        
        # Save plan JSON
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(results['plan'], f, indent=2, ensure_ascii=False)
        print(f"✓ Plan saved to: {plan_file}")
        
        # Save guideline JSON
        with open(guideline_file, 'w', encoding='utf-8') as f:
            json.dump(results['guideline'], f, indent=2, ensure_ascii=False)
        print(f"✓ Guideline saved to: {guideline_file}")
        
        print(f"\nReport preview:\n{results['report_markdown'][:500]}...")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
