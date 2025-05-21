# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

from redminelib import Redmine
from datetime import datetime
import os
import pandas as pd
import json

def scrap_wiki_pages(redmine_url: str, api_key: str, project_identifier: str, get_attachments: bool=False) -> pd.DataFrame:
    """
    Scrap all wiki pages for a given project and return metadata list.

    Returns a list of dicts with:
      - project_name                    
      - page_title
      - parent
      - full_hierarchy
      - status
      - created_on
      - updated_on
    """
    redmine = Redmine(redmine_url, key=api_key)
    project = redmine.project.get(project_identifier)
    wiki_pages = project.wiki_pages

    data = []
    for page in wiki_pages:
        full = redmine.wiki_page.get(resource_id=page.title, project_id=project.identifier)

        hierarchy = [full.title]
        parent = full.parent if hasattr(full, 'parent') else None
        while parent:
            hierarchy.insert(0, parent.title)
            try:
                parent = redmine.wiki_page.get(resource_id=parent.title, project_id=project.identifier).parent
            except Exception:
                break

        attachments_list =[] 
        print ("\n",full.title)


        if get_attachments:
            print ("\nWiki page attachments:\n")
            attachments = get_wiki_attachments(redmine,project_identifier,page)
            for att in attachments:
                print(att)
            
        data.append({
            'project_name': project.name,
            'project_identifier': project.identifier,
            'page_title': full.title,
            'parent': full.parent.title if hasattr(full, 'parent') else '',
            'full_hierarchy': "/".join(hierarchy),
            'status': 'default',
            'created_on': full.created_on.strftime("%Y-%m-%d %H:%M") if hasattr(full, 'created_on') else '',
            'updated_on': full.updated_on.strftime("%Y-%m-%d %H:%M") if hasattr(full, 'updated_on') else '',
            'attachments': attachments_list
        })

    df_wiki = pd.DataFrame.from_dict(data)

    return df_wiki

import mimetypes

def get_wiki_attachments(redmine, project_identifier:str, page) -> list:
    """
    Return attached files as a list, for each wiki page, with:
    - page_title 
    - file_name 
    - file_date : creation date
    - file_type : type (image, pdf, text, etc.)
    - file_size : size in Bytes 
    - download_url : URL to download the file
    """
    project = redmine.project.get(project_identifier)
    wiki_pages = project.wiki_pages

    attachments_info = []

    for page in wiki_pages:
        full = redmine.wiki_page.get(resource_id=page.title, project_id=project.identifier, include='attachments')
        if hasattr(full, 'attachments'):
            for att in full.attachments:                    
                mime_type, _ = mimetypes.guess_type(att.filename)
                file_type = mime_type.split('/')[0] if mime_type else 'unknown'
                attachments_info.append({
                    'page_title': full.title,
                    'file_name': att.filename,
                    'file_date': att.created_on.strftime("%Y-%m-%d %H:%M"),
                    'file_type': file_type,
                    'file_size':att.filesize,
                    'download_url': att.content_url
                })

    return attachments_info


def publish_wiki_table(redmine_url: str, api_key: str, project_identifier: str,
                       target_page: str, wiki_data: pd.DataFrame, comment: str = None):
    """
    Publish a wiki page containing a table of the provided wiki_data.

    wiki_data: list of dicts as returned by scrap_wiki_pages
    target_page: title for the new wiki page (identifier)
    comment: optional update comment
    """
    redmine = Redmine(redmine_url, key=api_key)
    project = redmine.project.get(project_identifier)

    lines = []
    header = ["Project name", "Project id", "Page title", "Link","Parent page", "Hierarchy", "Creation", "Latest update", "Status"]
    lines.append("| " + " |".join(f"*{h}*" for h in header) + " |")
    for _,row in wiki_data.iterrows():
        cells = [
            row['project_name'],
            row['project_identifier'],
            row['page_title'],
            "[["+row['project_name']+":"+ row['page_title']+"]]",
            row['parent'],
            row['full_hierarchy'],
            row['created_on'],
            row['updated_on'],
            row['status']

        ]
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines)

    try:
        wiki_page = redmine.wiki_page.get(resource_id=target_page, project_id=project.identifier)
        wiki_page.text = text
        wiki_page.comments = comment or f"Mis Ã  jour le {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        wiki_page.save()
        print(f"Page '{target_page}' updated")
    except:
        redmine.wiki_page.create(
            project_id=project.identifier,
            title=target_page,
            text=text,
            comments=comment or f"CrÃ©Ã© le {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"Page '{target_page}' created.")


from redminelib import Redmine
from datetime import datetime

def publish_wiki_attachments_table(redmine_url: str, api_key: str, project_identifier: str,
                                    target_page: str, wiki_data: pd.DataFrame, comment: str = None):
    """
    Publish a Redmine wiki page containing a table with one row per attachment.

    :param redmine_url: Redmine instance URL
    :param api_key: API key for authentication
    :param project_identifier: Redmine project identifier
    :param target_page: Title of the wiki page to create or update
    :param wiki_data: pandas.DataFrame with flattened attachment info, one row per attachment
    :param comment: Optional update comment
    """

    redmine = Redmine(redmine_url, key=api_key)
    project = redmine.project.get(project_identifier)

    lines = []
    header = ["Project", "Wiki Page", "Link", "File Name", "Date", "Type", "Size (KB)", "Download", "Status"]
    lines.append("| " + " |".join(f"*{h}*" for h in header) + " |")


    pages_with_attachment = wiki_data[wiki_data['attachments'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    attachments_exploded = wiki_data.explode("attachments").reset_index(drop=True)
    attachments_columns = pd.json_normalize(attachments_exploded['attachments'])
    
    combined_page_and_attach_metadata = pd.concat([pages_with_attachment, attachments_columns.drop(columns='attachments')], axis=1)
    print(combined_page_and_attach_metadata)
    # for _, row in wiki_data.iterrows():
    #     size_kb = f"{int(row['file_size']) // 1024}" if row.get('file_size') else ''
    #     download_link = f"[[{row['download_url']}|Download]]" if row.get('download_url') else ''
    #     cells = [
    #         row.get('project_name', ''),
    #         row.get('page_title', ''),
    #         row.get('link', ''),
    #         row.get('file_name', ''),
    #         row.get('file_date', ''),
    #         row.get('file_type', ''),
    #         size_kb,
    #         download_link,
    #         row.get('status', '')
    #     ]
    #     lines.append("| " + " |".join(cells) + " |")

    # text = "\n".join(lines)

    try:
        wiki_page = redmine.wiki_page.get(resource_id=target_page, project_id=project.identifier)
        wiki_page.text = text
        wiki_page.comments = comment or f"Updated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        wiki_page.save()
        print(f"Wiki page '{target_page}' updated.")
    except:
        redmine.wiki_page.create(
            project_id=project.identifier,
            title=target_page,
            text=text,
            comments=comment or f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"Wiki page '{target_page}' created.")


def extract_wiki_table_and_content_to_json(redmine_url: str, api_key: str,
                                            project_identifier: str, table_page: str,
                                            output_json_path: str):
    """
    Read a Redmine wiki page containing a metadata table and fetch corresponding wiki pages content.
    Save as JSON with content + metadata per page for RAG ingestion.

    The table must contain columns:
    *Projet* |*Title* |*Link* |*Parent page* |*Hierarchy* |*Creation* |*Latest update* |*Status*
    """
    redmine = Redmine(redmine_url, key=api_key)
    project = redmine.project.get(project_identifier)

    table_page_data = redmine.wiki_page.get(resource_id=table_page, project_id=project.identifier)
    lines = table_page_data.text.splitlines()

    documents = []
    headers = []
    for line in lines:
        if line.strip().startswith("|") and "*" in line:
            headers = [h.strip("* ").lower().replace(" ", "_") for h in line.strip("|").split("|")]
        elif line.strip().startswith("|") and headers:
            cells = [c.strip() for c in line.strip("|").split("|")]
            row = dict(zip(headers, cells))
            wiki_page_name = row.get("page_title", "").strip()

            project_identifier = row.get("project_id", "").strip()

            try:
                wiki_page = redmine.wiki_page.get(resource_id=wiki_page_name, project_id=project_identifier)
                content = wiki_page.text
            except Exception as e:
                content = f"[ERROR: Could not load page '{wiki_page_name}': {e}]"

            doc = {
                "content": content,
                "metadata": {
                    "project": row.get("project_name", ""),
                    "page_name": wiki_page_name,
                    "parent_page": row.get("parent_page", ""),
                    "hierarchy": row.get("hierarchy", ""),
                    "created_on": row.get("creation", ""),
                    "updated_on": row.get("latest_update", ""),
                    "status": row.get("status", "")
                }
            }
            documents.append(doc)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(documents)} entries to {output_json_path}")



if __name__ == "__main__":
    API_KEY = os.environ.get('MY_REDMINE_TOKEN')
    REDMINE_URL = os.environ.get('MY_REDMINE_URL')
    target_project = "sgsial"
    publish_project = "ops"
    publish_page= "tmp"
    publish_page2= "tmp2"
    output_json_path="wiki_pages_for_rag.json"

    # Scrap & publish
# wiki_data = scrap_wiki_pages(REDMINE_URL, API_KEY, target_project,get_attachments=False)
   # publish_wiki_table(REDMINE_URL, API_KEY, publish_project, publish_page, wiki_data)
   # publish_wiki_attachments_table(REDMINE_URL, API_KEY,publish_project, publish_page2, wiki_data) 

    extract_wiki_table_and_content_to_json(REDMINE_URL,API_KEY,publish_project,publish_page,output_json_path)
