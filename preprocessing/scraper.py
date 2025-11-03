import requests
import time
import random
import json
from bs4 import BeautifulSoup
from urllib.parse import quote

CATEGORY_URL = "https://en.wikipedia.org/wiki/Category:Rare_diseases"
BASE_URL = "https://en.wikipedia.org/wiki/"
disease_urls = []

# Canonical section names we want to collect
TARGET_SECTIONS = [
    'Signs and symptoms',
    'Causes',
    'Diagnosis',
    'Treatment',
    'Prognosis',
    'Epidemiology',
    'Society and culture',
]

# Common title variants per canonical section
SECTION_TITLE_VARIANTS = {
    'Signs and symptoms': ['Signs and symptoms', 'Signs_and_symptoms', 'Symptoms', 'Clinical features'],
    'Causes': ['Causes', 'Cause', 'Etiology', 'Aetiology'],
    'Diagnosis': ['Diagnosis', 'Diagnostic'],
    'Treatment': ['Treatment', 'Therapy', 'Management'],
    'Prognosis': ['Prognosis', 'Outcome'],
    'Epidemiology': ['Epidemiology', 'Epidemology', 'Prevalence'],  # include misspelling variant
    'Society and culture': ['Society and culture', 'Society_and_culture', 'Culture and society', 'Society', 'Culture'],
}

def fetch_category_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"Error fetching category page: {e}")
        return None


def extract_disease_urls(soup):
    global disease_urls
    category_div = soup.find('div', {'id': 'mw-pages'})
    
    if category_div:
        category_groups = category_div.find_all('div', {'class': 'mw-category-group'})
        
        for group in category_groups:
            links = group.find_all('a')
            for link in links:
                disease_name = link.get('title')
                href = link.get('href')
                
                if disease_name and href and '/wiki/Category:' not in href and '/wiki/' in href:
                    page_title = href.replace('/wiki/', '').replace('_', ' ')
                    # Exclude non-articles like "List of ..."
                    if disease_name.lower().startswith('list of'):
                        continue
                    # Use the page title derived from href to match the actual article path
                    candidate = page_title or disease_name
                    if candidate not in disease_urls:
                        disease_urls.append(candidate)
    
    next_page_link = soup.find('a', string='next page')
    if next_page_link:
        next_url = next_page_link.get('href')
        if next_url:
            return f"https://en.wikipedia.org{next_url}"
    return None
    

def fetch_wikipedia_data(disease):
    encoded_disease = disease.replace(' ', '_')
    url = BASE_URL + encoded_disease
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an error if the request failed
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"Error fetching {disease}: {e}")
        return None


def get_heading_level(tag):
    if not hasattr(tag, 'name'):
        return None
    if tag.name and tag.name.startswith('h') and len(tag.name) == 2 and tag.name[1].isdigit():
        return int(tag.name[1])
    return None


def find_heading_for_titles(soup, titles):
    """Find the first heading whose headline text matches any of the provided titles (case-insensitive)."""
    def norm(s):
        return ''.join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace()).replace('  ', ' ')
    normalized = [norm(t) for t in titles]

    # Prefer matching by the span.mw-headline id or text
    for hx in ['h2', 'h3', 'h4', 'h5', 'h6']:
        for heading in soup.find_all(hx):
            span = heading.find('span', class_='mw-headline')
            if span:
                span_id = norm(span.get('id') or '')
                span_text = norm(span.get_text())
                if span_id in normalized or span_text in normalized:
                    return heading

    # Fallback: contains match on heading text
    for heading in soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):
        text = norm(heading.get_text())
        for t in normalized:
            if t in text:
                return heading

    return None


def extract_section_with_subsections(heading):
    """Extract content after a heading, including content under deeper-level subheadings, stopping at the next heading of the same or higher level."""
    if not heading:
        return None

    # Handle wrapper divs like <div class="mw-heading mw-heading2"><h2 id="...">Title</h2> ...
    wrapper = heading.find_parent('div', class_='mw-heading')
    heading_node = heading
    if wrapper:
        # In some pages, the <hX> is inside the wrapper
        hx = wrapper.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if hx:
            heading_node = hx

    start_level = get_heading_level(heading_node) or 2
    parts = []

    # Start after the wrapper if present; otherwise after the heading
    current = (wrapper.next_sibling if wrapper is not None else heading.next_sibling)
    while current is not None:
        if hasattr(current, 'name'):
            # Stop when encountering a new section heading of same or higher level
            level = get_heading_level(current)
            if level is not None and level <= start_level:
                break
            # If we hit another wrapper div for a heading, check its level
            if current.name == 'div' and 'mw-heading' in current.get('class', []):
                next_h = current.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                next_level = get_heading_level(next_h) if next_h else None
                if next_level is not None and next_level <= start_level:
                    break

            # Capture text content
            if current.name == 'p':
                text = current.get_text().strip()
                if text:
                    parts.append(text)
        current = current.next_sibling

    return '\n'.join(parts) if parts else None


def extract_section_by_header(soup, titles):
    heading = find_heading_for_titles(soup, titles)
    content = extract_section_with_subsections(heading)
    if content:
        return content
    # Fallback: try older extractor bounded by next heading (more conservative)
    if heading:
        return extract_content_after_heading(heading)
    return None

def extract_content_after_heading(heading):
    """Extract all content following a heading until the next heading"""
    content_parts = []
    
    parent = heading.find_parent('div', class_='mw-heading')
    
    if parent:
        current = parent.next_sibling
    else:
        current = heading.next_sibling
    
    while current:
        if hasattr(current, 'name'):
            if current.name == 'div' and 'mw-heading' in current.get('class', []):
                break
            
            if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                break
            
            if current.name == 'p':
                text = current.get_text().strip()
                if text and len(text) > 0:
                    content_parts.append(text)
        
        current = current.next_sibling
    
    return '\n'.join(content_parts) if content_parts else None

# Helper function to find all sections in a Wikipedia page
def find_all_sections(soup):
    """Debug function to find all section headings and their IDs"""
    sections = []
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in headings:
        section_id = heading.get('id', '')
        section_text = heading.get_text().strip()
        
        span = heading.find('span', class_='mw-headline')
        if span:
            span_id = span.get('id', '')
            span_text = span.get_text().strip()
            section_id = span_id or section_id
            section_text = span_text or section_text
        
        if section_text:
            sections.append({'id': section_id, 'text': section_text})
    return sections

def save_to_json(data, filename="imprv_rare_diseases.json"):
    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def load_existing_json(filename="imprv_rare_diseases.json"):
    """Load existing JSON file if it exists"""
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def append_to_json(new_data, filename="imprv_rare_diseases.json"):
    """Append new data to existing JSON file"""
    existing_data = load_existing_json(filename)
    combined_data = existing_data + new_data
    save_to_json(combined_data, filename)
    return combined_data


def scrape_diseases(limit=200, output_filename="imprv_rare_diseases.json", offset=0, append=False):
    # Collect all disease URLs from all pages
    current_url = CATEGORY_URL
    page_count = 0
    
    print("Collecting all disease URLs from category pages...")
    
    while current_url:
        page_count += 1
        print(f"Fetching page {page_count}...")
        
        soup = fetch_category_page(current_url)
        if not soup:
            print(f"Failed to fetch page {page_count}.")
            break
        
        next_url = extract_disease_urls(soup)
        print(f"Found {len(disease_urls)} total diseases so far...")
        
        # Stop collecting if we reached enough to satisfy offset+limit
        if len(disease_urls) >= offset + limit:
            # Trim in case we overshot within a page
            del disease_urls[offset + limit:]
            current_url = None
            break
        
        if next_url:
            current_url = next_url
            time.sleep(random.uniform(0.5, 1))  # Small delay between page fetches
        else:
            current_url = None
    
    print(f"\nTotal diseases collected: {len(disease_urls)} from {page_count} pages.")
    
    # Now scrape each disease page
    disease_data = []

    start_index = min(offset, len(disease_urls))
    end_index = min(len(disease_urls), offset + limit)
    total_to_scrape = max(0, end_index - start_index)
    for idx, i in enumerate(range(start_index, end_index), start=1):
        disease_title = disease_urls[i]
        print(f"[{idx}/{total_to_scrape}] Scraping data for '{disease_title}'...")
        soup = fetch_wikipedia_data(disease_title)

        if not soup:
            time.sleep(random.uniform(1, 2))
            continue

        extracted = {}
        for canonical in TARGET_SECTIONS:
            titles = SECTION_TITLE_VARIANTS.get(canonical, [canonical])
            content = extract_section_by_header(soup, titles)
            if content:
                extracted[canonical] = content

        print(f"  -> matched sections: {list(extracted.keys())}")

        # Require at least four sections present
        if len(extracted) >= 4:
            record = {'Disease': disease_title}
            for canonical in TARGET_SECTIONS:
                if canonical in extracted:
                    record[canonical] = extracted[canonical]
            disease_data.append(record)

        time.sleep(random.uniform(1, 2))  # Delay between disease page fetches

    # Save results
    if append:
        append_to_json(disease_data, filename=output_filename)
    else:
        save_to_json(disease_data, filename=output_filename)
    print(f"\n{'='*80}")
    print(f"Successfully scraped {len(disease_data)} diseases from {page_count} category pages.")
    print(f"Data saved to '{output_filename}'.")
    print("="*80)


def scrape_single_disease(disease_title, output_filename="imprv_rare_diseases.json"):
    print(f"Scraping single article: '{disease_title}'...")
    soup = fetch_wikipedia_data(disease_title)
    if not soup:
        print("Failed to fetch article.")
        return []

    extracted = {}
    for canonical in TARGET_SECTIONS:
        titles = SECTION_TITLE_VARIANTS.get(canonical, [canonical])
        content = extract_section_by_header(soup, titles)
        if content:
            extracted[canonical] = content

    print(f"  -> matched sections: {list(extracted.keys())}")

    results = []
    if len(extracted) >= 4:
        record = {'Disease': disease_title}
        for canonical in TARGET_SECTIONS:
            if canonical in extracted:
                record[canonical] = extracted[canonical]
        results.append(record)
        save_to_json(results, filename=output_filename)
        print(f"Saved 1 record to '{output_filename}'.")
    else:
        print("Fewer than 4 required sections found; nothing saved.")

    return results

if __name__ == "__main__":
    scrape_diseases(limit=200, output_filename="imprv_rare_diseases.json")
    # pass
