import requests
import time
import random
import json
from bs4 import BeautifulSoup
from urllib.parse import quote

CATEGORY_URL = "https://en.wikipedia.org/wiki/Category:Rare_diseases"
BASE_URL = "https://en.wikipedia.org/wiki/"
disease_urls = []

def fetch_category_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
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
                    disease_urls.append(disease_name)
    
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
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error if the request failed
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"Error fetching {disease}: {e}")
        return None


def extract_section_by_header(soup, section_title):
    """Extract section content by matching section titles"""
    
    variations = [
        section_title,
        section_title.replace(' ', '_'),
        section_title.replace('_', ' '),
    ]

    
    for variation in variations:
        heading = soup.find(['h2', 'h3'], id=variation)
        if heading:
            return extract_content_after_heading(heading)
    
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for heading in headings:
        text = heading.get_text().strip().lower()
        if section_title.lower() in text:
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
            
            elif current.name in ['ul', 'ol']:
                for li in current.find_all('li'):
                    text = li.get_text().strip()
                    if text:
                        content_parts.append(f"• {text}")
        
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

def save_to_json(data, filename="rare_diseases.json"):
    with open(filename, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def load_existing_json(filename="rare_diseases.json"):
    """Load existing JSON file if it exists"""
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def append_to_json(new_data, filename="rare_diseases.json"):
    """Append new data to existing JSON file"""
    existing_data = load_existing_json(filename)
    combined_data = existing_data + new_data
    save_to_json(combined_data, filename)
    return combined_data


def scrape_diseases():
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
        
        if next_url:
            current_url = next_url
            time.sleep(random.uniform(0.5, 1))  # Small delay between page fetches
        else:
            current_url = None
    
    print(f"\nTotal diseases collected: {len(disease_urls)} from {page_count} pages.")
    
    # Now scrape each disease page
    disease_data = []
    
    for i in range(len(disease_urls)):
        test_disease = disease_urls[i]
        print(f"[{i+1}/{len(disease_urls)}] Scraping data for '{test_disease}'...")
        soup = fetch_wikipedia_data(test_disease)
        
        if soup:
            signs_symptoms = extract_section_by_header(soup, 'Signs_and_symptoms') or extract_section_by_header(soup, 'Signs and symptoms') or extract_section_by_header(soup, 'Symptoms')
            cause = extract_section_by_header(soup, 'Cause') or extract_section_by_header(soup, 'Causes')
            diagnosis = extract_section_by_header(soup, 'Diagnosis')
            treatment = extract_section_by_header(soup, 'Treatment')
            management = extract_section_by_header(soup, 'Management')
            
            disease_data.append({
                'Disease': test_disease,
                'Signs and symptoms': signs_symptoms or 'N/A',
                'Cause': cause or 'N/A',
                'Diagnosis': diagnosis or 'N/A',
                'Treatment': treatment or 'N/A',
                'Management': management or 'N/A'
            })
        
        time.sleep(random.uniform(1, 2))  # Delay between disease page fetches
    
    # Save to JSON file
    save_to_json(disease_data, filename="rare_diseases.json")
    print(f"\n{'='*80}")
    print(f"Successfully scraped {len(disease_data)} diseases from {page_count} category pages.")
    print(f"Data saved to 'rare_diseases.json'.")
    print("="*80)


scrape_diseases()
