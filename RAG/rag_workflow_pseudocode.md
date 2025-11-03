# Pseudocode: RAG Workflow for Rare Disease Article Generation

Notes from reading instruction_prompt_full.txt and article_structure.txt:
- Need to translate guidelines into concrete tasks for each section
- Check if we have enough info before generating article
- Filter sources by quality (no Wikipedia in retrieval, but can use for context)
- All claims need citations [n] format

## Main Workflow

```
def generate_rare_disease_article(disease_name):
    # Step 1: Break down guidelines into actionable to-dos
    todo_list = translate_guidelines_to_todos(disease_name)
    
    # Step 2: Actually retrieve info for each section
    retrieved_info = {}
    required_sections = get_all_sections()  # from article_structure.txt
    
    for section in required_sections:
        # Generate queries based on what this section needs
        queries = generate_queries_for_section(section, disease_name)
        retrieved_info[section] = retrieve_and_filter(queries, section)
    
    # Step 3: Do we have enough info to write this?
    sufficiency_report = assess_sufficiency(retrieved_info)
    
    # Step 4: Generate article if we have enough, otherwise return what we found
    if sufficiency_report.can_compose_article:
        article = generate_article(retrieved_info, sufficiency_report)
        return article, sufficiency_report
    else:
        # Maybe still useful to see what we got?
        return None, sufficiency_report
```

## Phase 1: Translate Guidelines to To-Do Items

This is where we take the article structure and break it down into what the agent actually needs to do.

```
def translate_guidelines_to_todos(disease_name):
    # These are the 13 sections from article_structure.txt
    required_sections = [
        "Title and Summary",
        "Quick Facts", 
        "Names and Classification",
        "What Happens in the Body",
        "Signs and Symptoms",
        "How It Is Diagnosed",
        "How Common Is It",
        "Causes and Genetics",
        "Treatment and Management",
        "Prognosis and Living With the Condition",
        "Research and Clinical Trials",
        "Patient Resources",
        "What Is Not Known"
    ]
    
    todo_list = []
    
    for section in required_sections:
        # What does this section actually need? (from article_structure.txt)
        requirements = extract_section_requirements(section)
        
        # Turn each requirement into a concrete task
        for req in requirements:
            todo = {
                "section": section,
                "task": req,
                "query_templates": generate_query_templates(req, disease_name),
                "min_sources": get_min_sources(req),  # some need more sources
                "citation_required": check_if_citation_needed(req)
            }
            todo_list.append(todo)
    
    return todo_list

def extract_section_requirements(section_name):
    # Basically mapping article_structure.txt to what we need to find
    # TODO: fill out all sections, only did a few as examples
    section_map = {
        "Quick Facts": [
            "prevalence/incidence with context",
            "typical age of onset",
            "main symptom cluster",
            "diagnostic test/criteria",
            "genetic or acquired classification",
            "treatment approach categories",
            "prognosis summary"
        ],
        "Signs and Symptoms": [
            "common symptoms with explanations",
            "less common symptoms",
            "red-flag symptoms for urgent care"
        ],
        "How It Is Diagnosed": [
            "clinical evaluation steps",
            "key tests and result meanings",
            "differential diagnosis (3-5 conditions)"
        ],
        # ... need to add the rest: Treatment, Genetics, etc.
        # Probably should load this from a config file or something
    }
    return section_map.get(section_name, [])
```

## Phase 2: RAG Retrieval with Quality Filtering

Here's where we actually go get the information. Need to filter out Wikipedia and low-quality sources.

```
def retrieve_and_filter(queries, section_name):
    all_documents = []
    
    for query in queries:
        # Search but exclude Wikipedia - per feedback, ban Wikipedia domain
        raw_docs = web_search(query, exclude_domains=["wikipedia.org", "en.wikipedia.org"])
        
        # Apply source quality filter based on instruction_prompt_full.txt hierarchy
        filtered_docs = filter_by_source_quality(raw_docs)
        
        all_documents.extend(filtered_docs)
    
    # Clean up duplicates and sort by quality
    unique_docs = deduplicate(all_documents)
    ranked_docs = rank_by_quality(unique_docs)  # best sources first
    
    return ranked_docs

def filter_by_source_quality(documents):
    # Quality hierarchy from instruction_prompt_full.txt lines 16-20
    # Tier 1 (best): peer-reviewed, guidelines
    # Tier 2: NIH/GARD/Orphanet, CDC/FDA, etc.
    # Tier 3: textbooks, specialty societies
    # Tier 4: Wikipedia (context only, but we're excluding in retrieval)
    # Exclude: social media, forums
    
    quality_scores = {
        "peer-reviewed reviews": 5,
        "guidelines, consensus": 5,
        "NIH/GARD/Orphanet/GeneReviews": 4,
        "CDC/FDA/EMA/Cochrane": 4,
        "medical textbooks": 3,
        "specialty society resources": 3,
        "wikipedia": 1,  # shouldn't get here since we exclude it
        "social media/forums": 0  # definitely exclude
    }
    
    filtered = []
    for doc in documents:
        score = get_source_quality_score(doc, quality_scores)
        # Only keep sources with score >= 3 (tier 3 and above)
        if score >= 3:
            doc.quality_score = score
            filtered.append(doc)
        # else: skip it
    
    return filtered
```

## Phase 3: Information Sufficiency Assessment

this sees if we have enough info to actually write a Wikipedia-style entry?

```
def assess_sufficiency(retrieved_info):
    # These sections are critical - need them all
    critical_sections = [
        "Quick Facts", 
        "What Happens in the Body", 
        "Signs and Symptoms", 
        "How It Is Diagnosed", 
        "Treatment and Management"
    ]
    
    section_scores = {}
    total_score = 0.0
    
    required_sections = get_all_sections()
    
    for section in required_sections:
        docs = retrieved_info.get(section, [])
        
        # Basic checks: do we have enough sources/content?
        num_sources = len(docs)
        word_count = sum(doc.word_count for doc in docs)
        num_citations = count_citations(docs)
        
        # Score this section (0.0 to 1.0)
        score = calculate_section_score(
            num_sources >= 3,      # at least 3 sources
            word_count >= 500,     # enough content
            num_citations >= 2,    # some citable facts
            section in critical_sections  # bonus if critical
        )
        
        section_scores[section] = {
            "score": score,
            "num_sources": num_sources,
            "word_count": word_count,
            "num_citations": num_citations,
            "sufficient": score >= 0.7  # threshold might need tuning
        }
        total_score += score
    
    # Overall decision: can we compose the article?
    # Need all critical sections + most other sections
    all_critical_good = all(
        section_scores[s]["sufficient"] 
        for s in critical_sections
    )
    
    num_sufficient = sum(
        1 for s in required_sections 
        if section_scores[s]["sufficient"]
    )
    
    # Probably need at least 10/13 sections, and all critical ones
    can_compose = (
        all_critical_good and 
        num_sufficient >= 10 and
        total_score >= 8.0  # rough threshold
    )
    
    return {
        "can_compose_article": can_compose,
        "section_scores": section_scores,
        "missing_info": identify_missing_info(section_scores),
        "gaps": identify_gaps(retrieved_info)  # for "What Is Not Known" section
    }

def calculate_section_score(has_sources, has_words, has_citations, is_critical):
    # Simple scoring - could be more sophisticated
    score = 0.0
    
    if has_sources:
        score += 0.3
    if has_words:
        score += 0.3
    if has_citations:
        score += 0.2
    if is_critical:
        score += 0.2  # extra weight for critical sections
    
    return min(score, 1.0)  # cap at 1.0
```

## Phase 4: Article Generation

If we have enough info, generate the actual article. Otherwise mark what's missing.

```
def generate_article(retrieved_info, sufficiency_report):
    article = initialize_article_template()  # from article_structure.txt
    
    required_sections = get_all_sections()
    
    for section in required_sections:
        section_data = sufficiency_report.section_scores[section]
        
        if section_data["sufficient"]:
            # We have enough info, generate the section
            content = rag_generate_section(
                section, 
                retrieved_info[section],
                get_section_instructions(section)  # from instruction_prompt_full.txt
            )
            article.add_section(section, content)
        else:
            # Not enough info - add placeholder
            article.add_section(section, "Evidence is limited for this section.")
            # Track what we don't know for section 12
            article.add_to_unknown_section(section)
    
    # Fix citation numbering across sections (make sure [1], [2], etc. are consistent)
    article = normalize_citations(article)
    
    # Run the quality checks from instruction_prompt_full.txt lines 140-147
    quality_report = run_quality_checks(article)
    
    # Fix any issues we found
    if not quality_report.passed:
        article = fix_quality_issues(article, quality_report)
    
    return article

def rag_generate_section(section, documents, instructions):
    # Build prompt combining section requirements + retrieved docs
    prompt = build_section_prompt(section, instructions, documents)
    
    # Generate using RAG agent (probably from HW1 notebook)
    # Need to ensure citations are in [n] format
    response = rag_agent.generate(
        prompt,
        citation_style="bracketed_numbers",  # [1], [2], etc.
        max_citations=len(documents)
    )
    
    return response

def run_quality_checks(article):
    # Self-check items from instruction_prompt_full.txt
    checks = {
        "has_safety_note": check_safety_note(article),  # must be in header
        "all_claims_cited": check_citations(article),     # every non-obvious claim
        "source_hierarchy": check_source_quality(article),  # no forums/social media
        "no_prescription": check_not_prescriptive(article),  # descriptive, not prescriptive
        "terms_defined": check_technical_terms(article),  # define on first use
        "numbers_consistent": check_number_consistency(article),  # no contradictions
        "no_speculation": check_no_speculation(article)  # only what sources say
    }
    
    passed = all(checks.values())
    return {"passed": passed, "checks": checks}
```

## Helper functions

Some extra utility functions

```
def generate_query_templates(requirement, disease_name):
    # Turn a requirement like "prevalence" into actual search queries
    # Should probably be smarter about this, but basic version:
    templates = [
        f"{disease_name} {requirement}",
        f"{requirement} {disease_name}",
        f"{disease_name} {requirement} GARD Orphanet",  # target high-quality sources
        f"{requirement} {disease_name} clinical guidelines"
    ]
    return templates

def get_min_sources(requirement):
    # Some things need more sources than others
    # Treatment/diagnosis are critical, so need more evidence
    if requirement in ["prevalence", "treatment", "diagnosis"]:
        return 3
    else:
        return 2  # default minimum

def check_if_citation_needed(requirement):
    # Per instruction_prompt_full.txt, most facts need citations
    # But simple lists like synonyms don't
    if requirement == "synonyms":
        return False
    # Probably most other things do
    return True
```

## Example Execution Flow

Quick ex with "AA amyloidosis":

```
# Input: disease_name = "AA amyloidosis"

1. translate_guidelines_to_todos("AA amyloidosis")
   → Breaks down into ~50+ specific tasks across 13 sections
   → Each task has query templates, min sources needed, etc.

2. retrieve_and_filter() for each section
   → "Quick Facts": run 5 queries → get 12 docs after filtering
   → "Treatment": run 4 queries → get 8 docs (filtered out Wikipedia, low-quality)
   → ... repeat for all 13 sections

3. assess_sufficiency(retrieved_info)
   → "Quick Facts": sufficient ✓ (3 sources, 800 words, 4 citations)
   → "Treatment": sufficient ✓ (5 sources, 1200 words, 6 citations)
   → "Patient Resources": insufficient ✗ (only 1 source, 200 words)
   → ... check all 13 sections
   → Overall: can_compose_article = True (11/13 sections sufficient, all critical ones good)

4. generate_article() if sufficient
   → Generate each section using RAG with retrieved docs
   → Normalize citations so they're [1], [2], [3]... across entire article
   → Run quality checks (safety note, citations, etc.)
   → Return final article + sufficiency report
```

Note: should handle edge cases (sparse entries) or if there's conflicting sources
