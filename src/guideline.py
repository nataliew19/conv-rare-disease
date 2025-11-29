"""
Guideline generation pipeline for rare disease Wikipedia-style articles.

This module generates comprehensive guidelines for writing articles about rare diseases
based on the disease name and desired Wikipedia entry style.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ArticleStyle(Enum):
    """Enumeration of supported Wikipedia article styles."""
    PATIENT_FACING = "patient_facing"  # Educational, layperson-friendly
    CLINICAL = "clinical"  # For healthcare professionals
    ACADEMIC = "academic"  # Research-oriented, scholarly
    STANDARD_WIKIPEDIA = "standard_wikipedia"  # Standard Wikipedia medical article style
    COMPREHENSIVE = "comprehensive"  # Detailed, exhaustive coverage


@dataclass
class Guideline:
    """Container for article generation guidelines.
    
    Attributes:
        disease_name: Name of the rare disease
        style: Style of Wikipedia entry
        audience: Target audience description
        sections: List of required sections to cover
        heuristics: List of heuristics/rules for writing
        citation_resources: Recommended resources for citations
        style_notes: Additional style-specific guidance
        citation_rules: Rules for how to cite sources
        quality_checklist: Checklist items for quality assurance
    """
    disease_name: str
    style: str
    audience: str
    sections: List[str] = field(default_factory=list)
    heuristics: List[str] = field(default_factory=list)
    citation_resources: List[str] = field(default_factory=list)
    style_notes: str = ""
    citation_rules: List[str] = field(default_factory=list)
    quality_checklist: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert guideline to dictionary."""
        return {
            "disease_name": self.disease_name,
            "style": self.style,
            "audience": self.audience,
            "sections": self.sections,
            "heuristics": self.heuristics,
            "citation_resources": self.citation_resources,
            "style_notes": self.style_notes,
            "citation_rules": self.citation_rules,
            "quality_checklist": self.quality_checklist,
        }
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"# Guideline for: {self.disease_name}",
            f"## Style: {self.style}",
            f"## Audience: {self.audience}",
            "",
            "## Required Sections:",
        ]
        for i, section in enumerate(self.sections, 1):
            lines.append(f"{i}. {section}")
        
        lines.extend([
            "",
            "## Heuristics:",
        ])
        for heuristic in self.heuristics:
            lines.append(f"- {heuristic}")
        
        lines.extend([
            "",
            "## Citation Resources (Priority Order):",
        ])
        for i, resource in enumerate(self.citation_resources, 1):
            lines.append(f"{i}. {resource}")
        
        if self.style_notes:
            lines.extend([
                "",
                "## Style Notes:",
                self.style_notes,
            ])
        
        if self.citation_rules:
            lines.extend([
                "",
                "## Citation Rules:",
            ])
            for rule in self.citation_rules:
                lines.append(f"- {rule}")
        
        if self.quality_checklist:
            lines.extend([
                "",
                "## Quality Checklist:",
            ])
            for item in self.quality_checklist:
                lines.append(f"- [ ] {item}")
        
        return "\n".join(lines)


def generate_guideline(
    rare_disease: str,
    style: str = "patient_facing",
    wikipedia_info: Optional[str] = None
) -> Guideline:
    """
    Generate a comprehensive guideline for writing a Wikipedia-style article about a rare disease.
    
    Args:
        rare_disease: Name of the rare disease
        style: Style of Wikipedia entry (patient_facing, clinical, academic, 
               standard_wikipedia, comprehensive)
        wikipedia_info: Optional extracted information from Wikipedia articles to enhance guidelines
    
    Returns:
        Guideline object with all necessary information
    """
    # Normalize style input
    style_lower = style.lower().replace(" ", "_").replace("-", "_")
    
    try:
        style_enum = ArticleStyle(style_lower)
    except ValueError:
        # Default to patient_facing if style not recognized
        style_enum = ArticleStyle.PATIENT_FACING
    
    # Base sections (from article_structure.txt)
    base_sections = [
        "Quick Facts",
        "Names and Classification",
        "What Happens in the Body",
        "Signs and Symptoms",
        "How It Is Diagnosed",
        "How Common Is It?",
        "Causes and Genetics",
        "Treatment and Management",
        "Prognosis and Living With the Condition",
        "Research and Clinical Trials",
        "Patient Resources",
        "What Is Not Known",
        "References",
    ]
    
    # Base citation resources (from instruction_prompt_full.txt)
    base_citation_resources = [
        "Peer-reviewed reviews, guidelines, consensus statements",
        "NIH or national resources (GARD, Orphanet, GeneReviews, CDC, FDA labels, EMA EPAR, Cochrane, ClinicalTrials.gov)",
        "Reputable medical textbooks or specialty society resources",
        "Wikipedia (context only, never as sole source for clinical facts)",
        "Exclude: social media, forums, non-curated blogs, non-reviewed preprints",
    ]
    
    # Style-specific customizations
    if style_enum == ArticleStyle.PATIENT_FACING:
        audience = "Informed layperson at 9th-10th grade reading level"
        heuristics = [
            "Use plain, respectful language. Short paragraphs. Active voice.",
            "Define technical terms in simple language at first use.",
            "Use concrete numbers with units and add quick comparisons when helpful.",
            "Be clear about what is known versus unknown.",
            "Avoid absolutes like 'cure' unless the source supports it.",
            "Neutral, supportive tone. No fear-inducing language.",
            "Include 'Important safety note' in header: 'This article is for education only and does not replace advice from your clinician.'",
            "No medical advice or speculation. Summarize what guidelines say, with citations.",
            "If data are limited, say so plainly and include 'What Is Not Known' section.",
        ]
        style_notes = """Writing style: Plain, respectful language at a high school level. 
Short paragraphs. Active voice. Define technical terms in simple language at first use. 
Use concrete numbers with units and add a quick comparison when helpful. 
Be clear about what is known versus unknown. Avoid absolutes like "cure" unless the source supports it. 
Use U.S. units and include metric in parentheses when relevant. 
Neutral, supportive tone. No fear-inducing language."""
        citation_rules = [
            "Use [1], [2], etc. in ascending order of first appearance.",
            "Place citation immediately after the claim it supports.",
            "Do not duplicate the same citation repeatedly in adjacent sentences unless claims differ.",
            "Do not include raw URLs in text. System will map numbers to bibliography separately.",
            "Cite every non-obvious claim to reliable sources.",
        ]
        quality_checklist = [
            "Include the safety note in the header",
            "All non-obvious claims cited with [n] right after the claim",
            "Sources consistent with hierarchy and free of forums or social posts",
            "Clearly mark uncertainties under 'What Is Not Known'",
            "Diagnosis and treatment sections descriptive and not prescriptive",
            "Technical terms defined on first use at 9th-10th grade level",
            "Numbers consistent and units included",
            "Avoid speculation or inference beyond sources",
        ]
    
    elif style_enum == ArticleStyle.CLINICAL:
        audience = "Healthcare professionals (physicians, nurses, medical students)"
        heuristics = [
            "Use appropriate medical terminology without excessive simplification.",
            "Include clinical decision-making considerations.",
            "Emphasize diagnostic criteria, differential diagnosis, and treatment protocols.",
            "Include relevant ICD-10 codes, diagnostic codes, and clinical guidelines.",
            "Reference evidence-based medicine principles and study quality.",
            "Discuss management strategies, monitoring parameters, and follow-up care.",
            "Include information about specialist referral indications.",
        ]
        style_notes = """Writing style: Professional medical writing. Use standard medical terminology. 
Include clinical decision-making frameworks. Reference evidence levels and study quality. 
Focus on practical clinical application."""
        citation_rules = [
            "Prioritize peer-reviewed clinical studies, systematic reviews, and clinical guidelines.",
            "Include study design, sample size, and evidence level when relevant.",
            "Use standard medical citation format.",
            "Reference clinical practice guidelines from recognized professional societies.",
        ]
        quality_checklist = [
            "All clinical claims supported by peer-reviewed evidence",
            "Diagnostic criteria clearly stated with source guidelines",
            "Treatment recommendations aligned with current clinical guidelines",
            "Differential diagnosis considerations included",
            "Monitoring and follow-up protocols specified",
        ]
        # Add clinical-specific sections
        base_sections.insert(5, "Clinical Presentation")
        base_sections.insert(7, "Differential Diagnosis")
        base_sections.insert(9, "Management Protocol")
    
    elif style_enum == ArticleStyle.ACADEMIC:
        audience = "Researchers, academics, and medical professionals seeking in-depth scientific information"
        heuristics = [
            "Include detailed methodology and study design information.",
            "Discuss research gaps and areas for future investigation.",
            "Reference primary research articles and recent publications.",
            "Include statistical data, confidence intervals, and study limitations.",
            "Discuss molecular mechanisms, genetic pathways, and pathophysiology in detail.",
            "Reference ongoing clinical trials and research directions.",
        ]
        style_notes = """Writing style: Scholarly, research-oriented. Include detailed scientific information. 
Discuss research methodology, study designs, and evidence quality. Reference primary sources. 
Include statistical measures and confidence intervals."""
        citation_rules = [
            "Prioritize primary research articles and recent peer-reviewed publications.",
            "Include study methodology, sample size, and statistical measures.",
            "Reference systematic reviews and meta-analyses when available.",
            "Cite original research and landmark studies.",
        ]
        quality_checklist = [
            "Primary research sources cited for key findings",
            "Research methodology and study quality discussed",
            "Statistical measures and confidence intervals included where relevant",
            "Research gaps and future directions clearly identified",
            "Molecular mechanisms and pathophysiology explained in detail",
        ]
        # Add academic-specific sections
        base_sections.insert(3, "Pathophysiology and Molecular Mechanisms")
        base_sections.insert(6, "Epidemiology and Research Data")
        base_sections.insert(10, "Current Research Directions")
    
    elif style_enum == ArticleStyle.STANDARD_WIKIPEDIA:
        audience = "General Wikipedia readers seeking comprehensive medical information"
        heuristics = [
            "Follow Wikipedia's medical article guidelines and Manual of Style.",
            "Maintain neutral point of view.",
            "Use reliable, verifiable sources that meet Wikipedia's notability requirements.",
            "Include infobox with key facts if applicable.",
            "Structure content hierarchically with clear headings.",
            "Include 'See also' and 'External links' sections where appropriate.",
            "Avoid original research and synthesis of primary sources.",
        ]
        style_notes = """Writing style: Wikipedia Manual of Style for medical articles. 
Neutral point of view. Verifiable, reliable sources. Hierarchical structure. 
No original research."""
        citation_rules = [
            "Use Wikipedia citation templates and format.",
            "All claims must be verifiable from reliable sources.",
            "Primary sources should be used sparingly; prefer secondary sources.",
            "Follow Wikipedia's reliable source guidelines for medical content.",
        ]
        quality_checklist = [
            "All content verifiable from reliable sources",
            "Neutral point of view maintained throughout",
            "No original research or synthesis",
            "Follows Wikipedia Manual of Style",
            "Infobox included if applicable",
        ]
        # Standard Wikipedia sections
        base_sections = [
            "Signs and symptoms",
            "Causes",
            "Diagnosis",
            "Treatment",
            "Prognosis",
            "Epidemiology",
            "History",
            "Society and culture",
            "Research",
            "References",
            "External links",
        ]
    
    else:  # COMPREHENSIVE
        audience = "Comprehensive audience: patients, families, healthcare professionals, and researchers"
        heuristics = [
            "Provide exhaustive coverage of all aspects of the disease.",
            "Include multiple perspectives: patient, clinical, and research viewpoints.",
            "Use layered complexity: start simple, add detail progressively.",
            "Include historical context, cultural aspects, and patient advocacy information.",
            "Cover all treatment modalities, including experimental and alternative approaches.",
            "Include detailed genetic information, inheritance patterns, and molecular mechanisms.",
            "Provide extensive resource lists and support organization information.",
        ]
        style_notes = """Writing style: Comprehensive, multi-layered. Start with accessible overview, 
then provide increasing detail. Include patient, clinical, and research perspectives. 
Cover historical, cultural, and advocacy aspects."""
        citation_rules = [
            "Use comprehensive citation strategy covering all source types.",
            "Include citations for historical, cultural, and advocacy information.",
            "Cite both primary research and authoritative secondary sources.",
            "Include patient advocacy and support organization resources.",
        ]
        quality_checklist = [
            "All major aspects of disease covered comprehensively",
            "Multiple perspectives included (patient, clinical, research)",
            "Historical and cultural context provided",
            "Extensive resource and support organization lists",
            "Both basic and advanced information included",
            "All claims properly cited",
        ]
        # Comprehensive sections include everything
        base_sections.insert(2, "Historical Context")
        base_sections.insert(8, "Alternative and Experimental Treatments")
        base_sections.insert(11, "Cultural and Social Aspects")
        base_sections.insert(12, "Advocacy and Support Organizations")
    
    # Enhance style_notes with Wikipedia information if provided
    if wikipedia_info:
        style_notes = f"{style_notes}\n\nWikipedia Analysis:\n{wikipedia_info}".strip()
    
    return Guideline(
        disease_name=rare_disease,
        style=style,
        audience=audience,
        sections=base_sections,
        heuristics=heuristics,
        citation_resources=base_citation_resources,
        style_notes=style_notes,
        citation_rules=citation_rules,
        quality_checklist=quality_checklist,
    )


def get_available_styles() -> List[str]:
    """Get list of available article styles."""
    return [style.value for style in ArticleStyle]


# ============================================================================
# Wikipedia Article Discovery and Extraction
# ============================================================================

def find_wikipedia_articles(disease: str, llm_client, llm_config) -> List[Dict[str, str]]:
    """Find related Wikipedia articles for the disease using LLM."""
    import json
    
    prompt = f"""Given the rare disease "{disease}", identify relevant Wikipedia articles.

Return a JSON list:
[
  {{"title": "Article Title", "url": "https://en.wikipedia.org/wiki/Article_Title", "reason": "Why relevant"}}
]

Focus on: main disease article, related conditions, treatments, genetics, patient resources.
Return only the JSON array."""
    
    response = llm_client.generate(
        prompt.strip(),
        model=llm_config.model_name,
        temperature=0.3,
        max_tokens=1000
    )
    
    # Parse JSON response
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) > 1:
            cleaned = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    
    try:
        articles = json.loads(cleaned)
        return articles if isinstance(articles, list) else []
    except json.JSONDecodeError:
        return []


def extract_wikipedia_info(articles: List[Dict[str, str]], disease: str, llm_client, llm_config) -> str:
    """Extract key information from Wikipedia articles to enhance guidelines."""
    if not articles:
        return ""
    
    articles_str = "\n".join([f"- {a.get('title', 'Unknown')}: {a.get('reason', '')}" for a in articles[:5]])
    
    prompt = f"""Analyze Wikipedia articles about "{disease}" to extract information for guideline generation.

Articles identified:
{articles_str}

Extract and summarize:
1. Key sections typically covered
2. Important terminology
3. Common citation sources
4. Structure patterns
5. Style guidelines

Provide a concise 2-3 paragraph summary."""
    
    return llm_client.generate(
        prompt.strip(),
        model=llm_config.model_name,
        temperature=0.4,
        max_tokens=800
    ).strip()


# Example usage and testing
if __name__ == "__main__":
    # Example: Generate guideline for Duchenne Muscular Dystrophy in patient-facing style
    guideline = generate_guideline("Duchenne Muscular Dystrophy", "patient_facing")
    print(guideline)
    print("\n" + "="*80 + "\n")
    
    # Example: Generate guideline in clinical style
    guideline_clinical = generate_guideline("Duchenne Muscular Dystrophy", "clinical")
    print(guideline_clinical)
    print("\n" + "="*80 + "\n")
    
    # Example: Show available styles
    print("Available styles:", get_available_styles())

