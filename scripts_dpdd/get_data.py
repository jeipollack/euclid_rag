#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv

def get_data_section_names(url, name):
    """
    Scrape a main page, find all subtopics, and print sections for each subtopic.
    
    Args:
        url (str): The URL of the main page containing subtopics
        name (str): Name identifier for the data source
    
    Returns:
        tuple: Two dictionaries:
            - h1_sections: Dictionary with h1 section names as keys and list of [subtopic, name, URL with anchor] lists as values
            - other_sections: Dictionary with other section names (h2, h3) as keys and list of [subtopic, name, URL with anchor] lists as values
    """
    # Initialize dictionaries to store sections
    h1_sections = {}
    other_sections = {}
    
    try:
        # Get the main page
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the main page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links that have class "reference internal"
        subtopic_links = soup.find_all('a', class_='reference internal', href=True)
        
        # print(f"\nProcessing {name} at {url}")
        # print("Found subtopics:")
        
        # Process each subtopic
        for link in subtopic_links:
            subtopic_url = link['href']
            subtopic_text = link.text.strip()
            # print(f"Link: {link}")
            # print(f"Text: {subtopic_text}")            
            
            # Handle relative URLs using urljoin
            if not subtopic_url.startswith(('http://', 'https://')):
                subtopic_url = urljoin(url, subtopic_url)
            
            # print(f"Original URL: {url}")
            # print(f"Subtopic URL: {subtopic_url}")
            
            try:
                # Get the subtopic page
                subtopic_response = requests.get(subtopic_url)
                subtopic_response.raise_for_status()
                
                # Parse the subtopic page
                subtopic_soup = BeautifulSoup(subtopic_response.text, 'html.parser')
                
                # Find all sections in the subtopic page
                sections = subtopic_soup.find_all(['section'])
                
                # print(f"\nSubtopic: {link.text.strip()}")
                # print("Sections found:")
                for section in sections:
                    # Get section name from the first heading in the section
                    section_name = section.find(['h1', 'h2', 'h3'])
                    if section_name:
                        section_text = section_name.text.lower().strip()
                        # print(f"- {section_text} => {section_name.name}")
                        
                        # Compute direct link to the section using its id
                        section_id = section.get('id') or section_name.get('id')
                        if section_id:
                            section_url_with_anchor = subtopic_url + '#' + section_id
                        else:
                            section_url_with_anchor = subtopic_url
                        
                        # Add to appropriate dictionary based on heading level
                        if section_name.name == 'h1':
                            if section_text not in h1_sections:
                                h1_sections[section_text] = []
                            h1_sections[section_text].append([subtopic_text, name, section_url_with_anchor])
                        else:
                            if section_text not in other_sections:
                                other_sections[section_text] = []
                            other_sections[section_text].append([subtopic_text, name, section_url_with_anchor])
                    
            except requests.RequestException as e:
                print(f"Error processing subtopic {subtopic_url}: {str(e)}")
            # print("____________________________")
                
    except requests.RequestException as e:
        print(f"Error accessing main page {url}: {str(e)}")
    
    return h1_sections, other_sections

# Scrape the content of a section URL
def scrape_section(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    h1 = soup.find("h1")
    if not h1:
        return soup.get_text(separator=" ", strip=True)
    texts = []
    for sib in h1.next_siblings:
        if getattr(sib, "name", None) == "h1":
            break
        if hasattr(sib, "get_text"):
            t = sib.get_text(separator=" ", strip=True)
            if t:
                texts.append(t)
    return " ".join(texts)

# Get full allowed sections with content
def get_data_section_full(url, name):
    # load allowed sections
    allowed_h1, allowed_other = set(), set()
    with open('h1_sections.csv', newline='') as f:
        r = csv.reader(f, delimiter=';'); next(r, None)
        for row in r:
            allowed_h1.add(row[0].strip().lower())
    with open('other_sections.csv', newline='') as f:
        r = csv.reader(f, delimiter=';'); next(r, None)
        for row in r:
            allowed_other.add(row[0].strip().lower())
    # get section names and and contents
    try:
        # Get the main page
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes        
        # Parse the main page
        soup = BeautifulSoup(response.text, 'html.parser')        
        # Find all links that have class "reference internal"
        subtopic_links = soup.find_all('a', class_='reference internal', href=True)

        results = []

        # Process each subtopic
        for link in subtopic_links:
            subtopic_url = link['href']
            subtopic_text = link.text.strip()
            
            # Handle relative URLs using urljoin
            if not subtopic_url.startswith(('http://', 'https://')):
                subtopic_url = urljoin(url, subtopic_url)
            
            try:
                # Get the subtopic page
                subtopic_response = requests.get(subtopic_url)
                subtopic_response.raise_for_status()
                
                # Parse the subtopic page
                subtopic_soup = BeautifulSoup(subtopic_response.text, 'html.parser')
                
                # Find all sections in the subtopic page
                sections = subtopic_soup.find_all(['section'])              
                for section in sections:
                    # Get section name from the first heading in the section
                    section_name = section.find(['h1', 'h2', 'h3'])
                    if section_name:
                        section_text = section_name.text.lower().strip()

                        if section_text not in allowed_h1 and section_text not in allowed_other:
                            continue

                        content = section.get_text(separator=" ", strip=True)
                        if content:
                            results.append({'content':content,'source':link,'section':section_text,'subtopic':subtopic_text,'source_name':name})
                        
                    
            except requests.RequestException as e:
                print(f"Error processing subtopic {subtopic_url}: {str(e)}")
            # print("____________________________")
                
    except requests.RequestException as e:
        print(f"Error accessing main page {url}: {str(e)}")

    return results