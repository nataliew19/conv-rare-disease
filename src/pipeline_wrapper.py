"""
Wrapper function to run the complete rare disease report generation pipeline.
Extracts the pipeline logic from gcp_rare_disease.ipynb into a reusable function.
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from textwrap import dedent
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path.cwd().resolve().parent / ".env", override=True)
load_dotenv("../.env", override=True)

# Import required modules
import sys
from pathlib import Path

# Add project root to path for imports (since src/src uses 'from src.' imports)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retriever_agent.serper_rm import SerperRM
from src.rag import RagAgent
from src.dataclass import RagResponse, RagRequest
from guideline import find_wikipedia_articles, extract_wikipedia_info, Guideline
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import tiktoken


class VertexLLMClient:
    """Vertex AI LLM client wrapper."""
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1", gcp_api_key: Optional[str] = None):
        if gcp_api_key and os.path.exists(gcp_api_key):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_api_key
            if not project_id:
                try:
                    with open(gcp_api_key, 'r') as f:
                        project_id = json.load(f).get("project_id")
                except (json.JSONDecodeError, IOError):
                    pass
        
        if not project_id:
            project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        
        if not project_id:
            raise ValueError("project_id is required. Set GCP_PROJECT_ID env var.")
        
        aiplatform.init(project=project_id, location=location)
        self.project_id = project_id

    def generate(self, prompt: str, *, model: str, temperature: float, max_tokens: int, **kwargs: Any) -> str:
        is_gemini = "gemini" in model.lower()
        
        if is_gemini:
            try:
                from vertexai.preview.generative_models import GenerativeModel
            except ImportError:
                from vertexai.generative_models import GenerativeModel
            
            model_variants = [model]
            if "gemini-2.5-pro" in model.lower():
                model_variants = ["gemini-2.5-pro", "gemini-2.0-flash-001", "gemini-1.5-pro-002"]
            elif "gemini-2.5-flash" in model.lower():
                model_variants = ["gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-1.5-flash-002"]
            elif "gemini-2.0-flash" in model.lower():
                model_variants = ["gemini-2.0-flash-001", "gemini-2.5-flash", "gemini-1.5-flash-002"]
            elif "gemini-1.5-pro" in model.lower():
                model_variants = ["gemini-1.5-pro-002", "gemini-1.5-pro", "gemini-2.0-flash-001"]
            elif "gemini-1.5-flash" in model.lower():
                model_variants = ["gemini-1.5-flash-002", "gemini-1.5-flash", "gemini-2.0-flash-001"]
            
            config = {"temperature": temperature, "max_output_tokens": max_tokens}
            if kwargs.get("reasoning_effort"):
                config["reasoning_effort"] = kwargs["reasoning_effort"]
            
            last_error = None
            for variant in model_variants:
                try:
                    return GenerativeModel(variant).generate_content(prompt, generation_config=config).text
                except Exception as e:
                    last_error = e
                    if "404" not in str(e) and "not found" not in str(e).lower():
                        raise
                    continue
            
            raise RuntimeError(f"None of the Gemini model variants {model_variants} are available. Last error: {last_error}") from last_error
        else:
            model_obj = aiplatform.TextGenerationModel.from_pretrained(model)
            params = {"temperature": temperature, "max_output_tokens": max_tokens}
            response = model_obj.predict(prompt, **params)
            return response.text if hasattr(response, "text") else str(response)


class LLMConfig:
    """Simple LLM config object."""
    def __init__(self, model_name: str = None, temperature: float = 0.4, max_tokens: int = 8192):
        self.model_name = model_name or os.getenv("VERTEX_MODEL", "gemini-1.5-pro-002")
        self.temperature = temperature
        self.max_tokens = max_tokens


class VertexEmbeddingEncoder:
    """Vertex embedding encoder with batching."""
    def __init__(self, model_name: str = "text-embedding-005"):
        self.model_name = model_name
        try:
            self.model = TextEmbeddingModel.from_pretrained(model_name)
        except Exception:
            self.model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            self.model_name = "textembedding-gecko@003"
        
        self.max_tokens_per_batch = 15000
        self.max_tokens_per_input = 20000
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def _truncate_text(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens_per_input:
            return text
        return self.tokenizer.decode(tokens[:self.max_tokens_per_input])
    
    def _batch_texts(self, texts):
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            truncated = self._truncate_text(text)
            tokens = len(self.tokenizer.encode(truncated))
            
            if current_tokens + tokens > self.max_tokens_per_batch:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [truncated]
                current_tokens = tokens
            else:
                current_batch.append(truncated)
                current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def aencode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        batches = self._batch_texts(texts)
        loop = asyncio.get_event_loop()
        all_embeddings = []
        
        for batch in batches:
            batch_tokens = sum(len(self.tokenizer.encode(t)) for t in batch)
            if batch_tokens > self.max_tokens_per_batch:
                mid = len(batch) // 2
                batch = batch[:mid]
            
            embeddings = await loop.run_in_executor(
                None,
                lambda b=batch: self.model.get_embeddings(b)
            )
            all_embeddings.extend([e.values for e in embeddings])
        
        return all_embeddings


def is_satisfied(articles, guideline, llm_client, llm_config, min_articles=3):
    """Check if current searches and guideline are satisfactory."""
    if len(articles) < min_articles:
        return False, "Not enough articles found"
    
    article_texts = " ".join([a.get('title', '') + " " + a.get('reason', '') for a in articles])
    article_texts_lower = article_texts.lower()
    
    key_areas = {
        'main_disease': any(term in article_texts_lower for term in ['disease', 'disorder', 'syndrome']),
        'symptoms': any(term in article_texts_lower for term in ['symptom', 'sign', 'clinical', 'manifestation']),
        'treatment': any(term in article_texts_lower for term in ['treatment', 'therapy', 'drug', 'medication']),
        'genetics': any(term in article_texts_lower for term in ['genetic', 'gene', 'mutation', 'inheritance']),
        'diagnosis': any(term in article_texts_lower for term in ['diagnosis', 'diagnostic', 'test', 'testing'])
    }
    
    coverage_score = sum(key_areas.values()) / len(key_areas)
    
    if coverage_score >= 0.6 and guideline is not None:
        prompt = f"""Evaluate if this guideline is comprehensive enough to generate a research plan.

Guideline sections: {len(guideline.sections)}
Heuristics: {len(guideline.heuristics)}
Citation resources: {len(guideline.citation_resources)}

Key areas covered: {', '.join([k for k, v in key_areas.items() if v])}

Respond with ONLY "YES" if comprehensive enough, or "NO" with a brief reason if it needs more information."""
        
        evaluation = llm_client.generate(
            prompt.strip(),
            model=llm_config.model_name,
            temperature=0.2,
            max_tokens=100
        ).strip()
        
        if evaluation.upper().startswith("YES"):
            return True, f"Comprehensive coverage ({coverage_score:.1%})"
        else:
            reason = evaluation.replace("NO", "").strip() if "NO" in evaluation.upper() else "Needs more information"
            return False, f"Coverage: {coverage_score:.1%}, {reason}"
    
    if coverage_score >= 0.7:
        return True, f"Good coverage ({coverage_score:.1%})"
    else:
        missing = [k for k, v in key_areas.items() if not v]
        return False, f"Coverage: {coverage_score:.1%}, missing: {', '.join(missing)}"


def generate_next_search_query(disease_name, current_articles, guideline, llm_client, llm_config):
    """Generate next search query to improve guideline coverage."""
    current_titles = [a.get('title', '') for a in current_articles]
    current_titles_str = ", ".join(current_titles[:5])
    
    prompt = f"""Given the disease "{disease_name}" and current Wikipedia articles found:
{current_titles_str}

Generate a focused search query to find additional Wikipedia articles that would help improve guideline coverage.
Focus on gaps in: treatments, genetics, patient resources, related conditions, or research.

Return only a single search query string (no JSON, no explanation)."""
    
    query = llm_client.generate(
        prompt.strip(),
        model=llm_config.model_name,
        temperature=0.5,
        max_tokens=50
    ).strip()
    
    return query.strip('"').strip("'").strip()


def run_pipeline(
    disease_name: str,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run the complete rare disease report generation pipeline.
    
    Args:
        disease_name: Name of the rare disease
        output_dir: Directory to save outputs (defaults to src/output)
        progress_callback: Optional callback function(status_message) for progress updates
    
    Returns:
        Dictionary with keys: 'guideline', 'rag_response', 'report', 'output_paths'
    """
    if progress_callback:
        progress_callback("Initializing pipeline...")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("src/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize clients
    project_id = os.getenv("GCP_PROJECT_ID")
    gcp_api_key = os.getenv("GCP_API_KEY")
    vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")
    vertex_model = os.getenv("VERTEX_MODEL", "gemini-1.5-pro-002")
    serper_api_key = os.getenv("SERPER_API_KEY")
    
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable not set")
    if not serper_api_key:
        raise ValueError("SERPER_API_KEY environment variable not set")
    
    llm_client = VertexLLMClient(project_id=project_id, location=vertex_location, gcp_api_key=gcp_api_key)
    llm_config = LLMConfig(model_name=vertex_model)
    
    # Step 1: Wikipedia Discovery & Guideline Generation
    if progress_callback:
        progress_callback(f"🔍 Starting Wikipedia discovery for: {disease_name}")
    
    all_articles = []
    wikipedia_info = ""
    guideline = None
    max_iterations = 3
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        if progress_callback:
            progress_callback(f"Wikipedia discovery iteration {iteration}/{max_iterations}")
        
        if iteration == 1:
            search_query = disease_name
        else:
            search_query = generate_next_search_query(disease_name, all_articles, guideline, llm_client, llm_config)
        
        new_articles = find_wikipedia_articles(search_query, llm_client, llm_config)
        existing_titles = {a.get('title', '').lower() for a in all_articles}
        unique_new = [a for a in new_articles if a.get('title', '').lower() not in existing_titles]
        all_articles.extend(unique_new)
        
        if progress_callback:
            progress_callback(f"Found {len(unique_new)} new articles (total: {len(all_articles)})")
        
        wikipedia_info = extract_wikipedia_info(all_articles, disease_name, llm_client, llm_config)
        
        # Generate guideline
        prompt = f"""Generate a comprehensive guideline for writing a Wikipedia-style article about "{disease_name}".

Style: patient_facing (for educated lay readers at 9th-10th grade reading level)

Wikipedia information gathered:
{wikipedia_info if wikipedia_info else "No Wikipedia information available yet."}

Create a guideline with:
1. Required sections (list of section names)
2. Writing heuristics (rules for writing style)
3. Citation resources (priority order)
4. Citation rules (how to cite)
5. Quality checklist

Return as JSON with keys: sections (list), heuristics (list), citation_resources (list), citation_rules (list), quality_checklist (list), style_notes (string), audience (string)."""
        
        response = llm_client.generate(
            prompt.strip(),
            model=llm_config.model_name,
            temperature=0.4,
            max_tokens=2000
        )
        
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if len(lines) > 1:
                cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        
        try:
            guideline_dict = json.loads(cleaned)
            guideline = Guideline(
                disease_name=disease_name,
                style="patient_facing",
                audience=guideline_dict.get("audience", "Informed layperson at 9th-10th grade reading level"),
                sections=guideline_dict.get("sections", []),
                heuristics=guideline_dict.get("heuristics", []),
                citation_resources=guideline_dict.get("citation_resources", []),
                citation_rules=guideline_dict.get("citation_rules", []),
                quality_checklist=guideline_dict.get("quality_checklist", []),
                style_notes=guideline_dict.get("style_notes", "")
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: create minimal guideline
            guideline = Guideline(
                disease_name=disease_name,
                style="patient_facing",
                audience="Informed layperson at 9th-10th grade reading level",
                sections=["Quick Facts", "Signs and Symptoms", "Treatment", "References"],
                heuristics=["Use plain language", "Cite all claims"],
                citation_resources=["Peer-reviewed sources"],
                citation_rules=["Use numeric citations"],
                quality_checklist=["All claims cited"]
            )
        
        satisfied, reason = is_satisfied(all_articles, guideline, llm_client, llm_config)
        if satisfied:
            break
    
    # Save guideline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    disease_safe = disease_name.replace(" ", "_").replace("/", "_").lower()
    guideline_json_path = output_dir / f"guideline_{disease_safe}_{timestamp}.json"
    guideline_md_path = output_dir / f"guideline_{disease_safe}_{timestamp}.md"
    
    with open(guideline_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "guideline": guideline.to_dict(),
            "metadata": {
                "disease_name": disease_name,
                "articles_found": len(all_articles),
                "articles": all_articles,
                "wikipedia_info": wikipedia_info,
                "iterations": iteration,
                "timestamp": timestamp
            }
        }, f, indent=2, ensure_ascii=False)
    
    with open(guideline_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Guideline for: {disease_name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Articles found:** {len(all_articles)}\n\n")
        f.write(str(guideline))
    
    if progress_callback:
        progress_callback("✅ Guideline generated")
    
    # Step 2: RAG Agent Run
    if progress_callback:
        progress_callback("🔍 Running RAG agent to gather evidence...")
    
    encoder = VertexEmbeddingEncoder(model_name="text-embedding-005")
    serper_retriever = SerperRM(api_key=serper_api_key, encoder=encoder)
    
    # Create DSPy LM wrapper
    import dspy
    from types import SimpleNamespace
    
    class VertexDSPyLM(dspy.BaseLM):
        def __init__(self, vertex_client, model_name, temperature=0.3, max_tokens=2048):
            super().__init__(model=model_name, temperature=temperature)
            self.vertex_client = vertex_client
            self.default_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        
        def __call__(self, prompt, **kwargs):
            cfg = {**self.default_kwargs, **kwargs}
            return self.vertex_client.generate(
                prompt,
                model=self.model,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            )
        
        async def aforward(self, prompt=None, messages=None, **kwargs):
            cfg = {**self.default_kwargs, **kwargs}
            if messages:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: self.vertex_client.generate(
                    prompt,
                    model=self.model,
                    temperature=cfg["temperature"],
                    max_tokens=cfg["max_tokens"],
                ),
            )
            
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(role="assistant", content=text),
                    finish_reason="stop",
                )],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                model=self.model,
            )
    
    vertex_rag_lm = VertexDSPyLM(llm_client, model_name=vertex_model)
    rag_agent = RagAgent(retriever=serper_retriever, rag_lm=vertex_rag_lm)
    
    rag_request = RagRequest(
        question=f"Provide an informative entry about {disease_name}.",
        max_retriever_calls=3,
    )
    
    rag_response = asyncio.run(rag_agent.aforward(rag_request))
    
    rag_response_path = output_dir / "rag_response.json"
    with open(rag_response_path, "w") as f:
        json.dump(rag_response.to_dict(), f, indent=2)
    
    if progress_callback:
        progress_callback("✅ RAG evidence gathered")
    
    # Step 3: Generate Research Plan and Report
    if progress_callback:
        progress_callback("📝 Generating research plan...")
    
    topic = f"Informative and Accurate Entry of {disease_name}, describing the rare disease, its symptoms and suspected causes, and potential treatments or current efforts to find treatments or self treatment IF available."
    guideline_text = str(guideline)
    
    # Format sources with full details for proper citation
    sources_list = []
    for i, doc in enumerate(rag_response.cited_documents, 1):
        # Build full reference information
        source_parts = []
        if doc.title:
            source_parts.append(f"Title: {doc.title}")
        source_parts.append(f"URL: {doc.url}")
        if doc.timestamp:
            if hasattr(doc.timestamp, 'strftime'):
                access_date = doc.timestamp.strftime('%Y-%m-%d')
            else:
                access_date = str(doc.timestamp)
            source_parts.append(f"Accessed: {access_date}")
        elif doc.metadata and 'access_date' in doc.metadata:
            source_parts.append(f"Accessed: {doc.metadata['access_date']}")
        else:
            source_parts.append(f"Accessed: {datetime.now().strftime('%Y-%m-%d')}")
        
        if doc.metadata:
            if 'author' in doc.metadata:
                source_parts.append(f"Author: {doc.metadata['author']}")
            if 'publication_date' in doc.metadata:
                source_parts.append(f"Publication Date: {doc.metadata['publication_date']}")
            if 'publisher' in doc.metadata:
                source_parts.append(f"Publisher: {doc.metadata['publisher']}")
        
        source_info = f"[{i}] " + " | ".join(source_parts)
        sources_list.append(source_info)
    
    sources_text = "\n".join(sources_list)
    num_sources = len(rag_response.cited_documents)
    
    # Format RAG evidence similar to notebook format
    cited = ", ".join(f"[{i+1}] {doc.url}" for i, doc in enumerate(rag_response.cited_documents))
    rag_evidence_text = f"""Q: {rag_response.question}

A: {rag_response.answer}

Sources: {cited}

Available Sources (Total: {num_sources} sources - use ALL of these citations throughout the report):
{sources_text}

IMPORTANT CITATION INSTRUCTIONS:
- The RAG answer above contains citations [1], [2], [3], etc. that refer to the sources listed above
- Use MULTIPLE different sources throughout your report - do not rely on just one source
- Different sections should cite different sources to provide comprehensive coverage
- Use the citation numbers [1], [2], [3], etc. that appear in the RAG answer above
- Do NOT invent new citations or use placeholder formats like "[Author Year]"
- Ensure you cite from at least {min(3, num_sources)} different sources across different sections of the report"""
    
    plan_prompt = dedent(f"""
    Task: Create a step-by-step research plan for **{topic}**.

    Guideline to follow:
    {guideline_text}

    Evidence gathered via RAG:
    {rag_evidence_text}
    
    When creating the plan:
    - Reference sources using the numeric citations [1], [2], [3], etc. that appear in the RAG answer above
    - Plan to use MULTIPLE different sources across different sections - you have {num_sources} sources available
    - Ensure the plan incorporates information from different sources to provide comprehensive coverage
    """).strip()
    
    plan_system = "You are a meticulous medical editor. Output numbered steps with citations."
    plan_result = llm_client.generate(
        plan_prompt,
        model=vertex_model,
        temperature=0.8,
        max_tokens=8192
    )
    
    if progress_callback:
        progress_callback("📄 Generating final report...")
    
    synth_prompt = dedent(f"""
    Task: Using the plan below, produce a hierarchical research report for **{disease_name}** suitable to transform into a Wikipedia-style article.

    FORMAT: Write in prose/paragraph format like a Wikipedia article - use full sentences and paragraphs, NOT bullet points. Each section should contain flowing text with inline citations using the numeric format [1], [2], [3], etc. that correspond to the sources provided in the RAG evidence.
    Clearly separate **Facts**, **Uncertainties**, and **Controversies**. 
    
    CITATION REQUIREMENTS:
    - Use MULTIPLE different sources throughout the report - you have {num_sources} sources available
    - In the body: Use numeric citations [1], [2], [3], etc. after facts/claims - cite different sources for different facts
    - Distribute citations across sections: different sections should cite different sources
    - In the References section (section 13): Provide FULL references with complete information (title, URL, access date) for each numbered citation
    - Use citation format [1][2] when multiple sources support the same claim
    
    CRITICAL: 
    - Use ONLY the citation numbers [1], [2], [3], etc. from the RAG evidence provided below
    - Do NOT invent citations or use placeholder formats like "[Author Year]" or "[Clinical Study A, Year]"
    - Use at least {min(3, num_sources)} different sources across the report - do not rely on just one source
    - The RAG answer contains citations [1], [2], etc. - use these same citation numbers in your report

    Plan to execute:
    {plan_result}

    RAG Evidence with Sources:
    {rag_evidence_text}

    Output structure (each of the following should be its own section, NOT a bullet point):
    - Lead / Summary (2-3 paragraphs with citations, NOT bullets)
    - Overview / Definition
    - Signs & Symptoms
    - Genetics & Pathophysiology
    - Diagnosis
    - Management / Treatment
    - Prognosis / Natural History
    - Epidemiology
    - History / Nomenclature
    - Research & Trials
    - Tables
      - T1. Genotype–Phenotype Mapping (Gene | Variant class | Key features | Penetrance | Source)
      - T2. Differential Diagnosis (Condition | Distinguishing features | Test | Source)
      - T3. Management Summary (Issue | Recommendation | Evidence level | Source)
    - Uncertainties & Controversies
    - References
      Provide full, properly formatted references for each source cited in the report. 
      
      FORMAT REQUIREMENTS:
      - Each reference must be on a NEW LINE
      - Each reference must start with [index] followed by a space
      - Format: [1] Full Title (if available). Full URL. Accessed: [date if available, or use current date format YYYY-MM-DD]
      - Format: [2] Full Title (if available). Full URL. Accessed: [date if available, or use current date format YYYY-MM-DD]
      - Continue this pattern for all sources
      
      EXAMPLE FORMAT:
      [1] Fabry Disease: Symptoms & Causes - Cleveland Clinic. https://my.clevelandclinic.org/health/diseases/16235-fabry-disease. Accessed: 2025-11-30
      [2] Fabry's Disease | Cedars-Sinai. https://www.cedars-sinai.org/health-library/diseases-and-conditions/f/fabrys-disease.html. Accessed: 2025-11-30
      [3] Fabry Disease - Symptoms, Causes, Treatment | NORD. https://rarediseases.org/rare-diseases/fabry-disease/. Accessed: 2025-11-30
      
      Include ALL sources that were cited in the report (using the numbers [1], [2], etc.). Format them as complete, professional references with all available information. Use the exact sources from the RAG evidence provided above - do not invent or add sources that were not in the RAG evidence.
    """).strip()
    
    synth_system = "You are a senior biomedical writer. Follow the outline exactly and cite only the provided evidence."
    report_result = llm_client.generate(
        synth_prompt,
        model=vertex_model,
        temperature=0.8,
        max_tokens=8192
    )
    
    # Save final report
    report_path = output_dir / f"report_{disease_safe}_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_result)
    
    if progress_callback:
        progress_callback("✅ Report generated!")
    
    return {
        "guideline": guideline.to_dict(),
        "rag_response": rag_response.to_dict(),
        "report": report_result,
        "metadata": {
            "disease_name": disease_name,
            "articles_found": len(all_articles),
            "iterations": iteration,
            "timestamp": timestamp
        },
        "output_paths": {
            "guideline_json": str(guideline_json_path),
            "guideline_md": str(guideline_md_path),
            "rag_response": str(rag_response_path),
            "report": str(report_path)
        }
    }

