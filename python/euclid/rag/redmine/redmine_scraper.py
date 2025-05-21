import mimetypes
from redminelib import Redmine
from datetime import datetime
import os
import pandas as pd
import json


some_projects_1 = {"LE2_PFs" :  {"PHZ": "phz_pf"}}
some_projects_2 = {"SGS Developer data" :  {"SGS coding standards": "coding-standards"}}
some_projects_3 = {"SGS Developer data" :  {"tmp": "parseword"}}


sgs_projects = {

		# List of all the SGS PFs    
		"LE1_PFs" : {"LE1": "le1pf"},
		"LE2_PFs" : {"NIR": "nir_pf", "VIS": "vis_pf", "SPE": "spe_pf", "MER": "mer_pf", "SHE": "she_pf", "PHZ": "phz_pf",
					 "SIR": "sir_pf", "SIM": "sim_pf", "EXT": "sgu"},
  
  
}
tmp = {

		"LE3_Weak_Lensing" : {"2PCF-WL": "2PCF-WL", "CM-2PCF-WL": "CM-2PCF-WL", "PK-WL": "PK-WL", "CM-PK-WL": "CM-PK-WL",
							"2D-MASS-WL": "2D-MASS-WL"},

		"LE3_Galaxy_Clustering" : {"2PCF-GC": "2PCF-GC", "PK-GC": "PK-GC", "CM-2PCF-GC": "CM-2PCF-GC", "CM-PK-GC": "CM-PK-GC",
								 "3PCF-GC": "3PCF-GC", "BK-GC": "BK-GC"},

		"LE3_Cluster_of_Galaxies" : {"DET-CL": "DET-CL", "SET-CL": "SET-CL", "RICH-CL": "RICH-CL", "Z-CL": "Z-CL",
								   "SIGV-CL": "SIGV-CL",
								   "COMB-CL": "COMB-CL", "LMF-CL": "LMF-CL", "PROF-CL": "PROF-CL", "CAT-CL": "CAT-CL",
								   "2PCF-CL-CL": "2PCF-CL-CL",
								   "PK-CL": "PK-CL", "3PCF-CL": "3PCF-CL", "BK-CL": "BK-C", "CM-2PCF-CL": "CM-2PCF-CL",
								   "CM-PK-CL": "CM-PK-CL" },

		"LE3_Internal_Data" : {"VMPZ-ID": "VMPZ-ID", "VMSP-ID": "VMSP-ID", "SEL-ID": "SEL-ID", "LMF-ID": "LMF-ID"},

		"LE3_External_Data" : {"MGC-ED": "MGC-ED", "MCC-ED": "MCC-ED", "GALEXT-ED": "GALEXT-ED"},

		"LE3_Milky_Way_Nearby_Galaxies" : {"NG-MWNG": "NG-MWNG", "SP-MWNG": "SP-MWNG", "RSS-MWNG": "RSS-MWNG"},

		"LE3_Time_Domain" : {"TCRL-TD": "TCRL-TD"},
  
  		"System_team" : {"ST": "sgsst", "SGS Architecture":"sub2pw","Common Deployment System":"codeps",
                     "Common Tools":"sgssu","Common Orchestration System (COORS)":"",
                     "Data Model":"","Data Quality":"","Common Tools":"","Euclid Archive System":"",
                     "IAL":"sgsial","Mock-up":"","MonitoringAndControl":"","SGS Development Pages":"","SGS Integration":"",
                     "SGS system administration":""}
    
}
                     
                     



class RedmineScraper:
    def __init__(self,
                 redmine_projects_to_browse:dict, 
                 publish_project:str,
                 get_wiki_pages:bool=True,
                 get_attachments:bool=False):
        
        api_key = os.environ.get('MY_REDMINE_TOKEN')
        redmine_url = os.environ.get('MY_REDMINE_URL')
        self.redmine = Redmine(redmine_url, key=api_key)
        
        self.get_wiki_pages = get_wiki_pages
        self.get_attachments = get_attachments
        self.extract_content_to_json = True
        
        
        self.redmine_projects_to_browse = redmine_projects_to_browse
        self.publish_project = publish_project
        self.input_rag_wiki_pages_parent= "Rag_wiki_pages"
        self.input_rag_attached_files_parent= "Rag_attached_files"
        self.output_attached_json_filename="rag_attached"
        self.output_wiki_json_filename="rag_wiki"

    def get_wiki_page(self, project_identifier: str, page_title: str):
        project = self.redmine.project.get(project_identifier)
        wiki_page = self.redmine.wiki_page.get(resource_id=page_title, project_id=project.identifier)
        return wiki_page
    
    
    def get_wiki_attachments(self, project_identifier: str, page_title: str):
        project = self.redmine.project.get(project_identifier)
        wiki_page = self.redmine.wiki_page.get(resource_id=page_title, project_id=project.identifier, include='attachments')
        attachments = wiki_page.attachments if hasattr(wiki_page, 'attachments') else []
        return attachments
    
    def scrap_and_publish_redmine_projects(self):
        
    
        for domain in self.redmine_projects_to_browse:
            print(f"Browsing domain: {domain}")
            
            for title,project in self.redmine_projects_to_browse[domain].items():            

                print(f"Scraping project: {project}")
                df_wiki = self.scrap_project_wiki_pages(project)
                print(f"Scraped {len(df_wiki)} wiki pages for project {project}")
                publish_page = f"Wiki_pages_from_{project}"
                self.publish_project_wiki_table(project, publish_page, df_wiki)
                print(f"Published wiki page table for project {project} to {publish_page}")
                    
                if self.get_attachments:
                    print(f"Scraping attachments for project: {project}")
                    publish_page = f"Attachments_from_{project}"
                    self.publish_wiki_attachements_table(project, publish_page, df_wiki)
                    print(f"Published wiki attachments table for project {project} to {publish_page}")
                    
                    
                    
                if self.extract_content_to_json:
                    print(f"Extracting wiki table and content to JSON for project: {project}")
                    output_json_path = self.extract_wiki_table_and_content_to_json(project,publish_page)
                    #print(f"Extracted wiki table and content to JSON for project {project} to {output_json_path}")

                self.extract_wiki_table_and_content_to_json
            
        


    def scrap_project_wiki_pages(self,project_identifier: str) -> pd.DataFrame:
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
        project = self.redmine.project.get(project_identifier)
        wiki_pages = project.wiki_pages

        data = []
        for page in wiki_pages:
            full = self.redmine.wiki_page.get(resource_id=page.title, project_id=project.identifier)

            hierarchy = [full.title]
            parent = full.parent if hasattr(full, 'parent') else None
            while parent:
                hierarchy.insert(0, parent.title)
                try:
                    parent = self.redmine.wiki_page.get(resource_id=parent.title, project_id=project.identifier).parent
                except Exception:
                    break

            attachments_list =[] 
            print ("\n",full.title)


            if self.get_attachments:
                print ("\nWiki page attachments:\n")
                attachments_list = self.get_wiki_attachments(project_identifier,page)
                for att in attachments_list:
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
                'wiki_category': self.categorize_filename(full.title),
                'attachments': attachments_list
            })

        df_wiki = pd.DataFrame.from_dict(data)

        return df_wiki


    def categorize_filename(sel,filename: str) -> str:
        filename_lower = filename.lower()

        categories = {
            "meeting_minutes": ["mom", "meeting", "minutes", "réunion"],
            "pf_document": ["rsd", "sdd", "sts", "str","srn", "sum"],
            "plan": ["plan", "planning", "roadmap"],
            "default": []  # fallback
        }

        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category

        return "default"


    def get_wiki_attachments(self,project_identifier:str, page) -> list:
        """
        Retourne la liste des fichiers attaches Ã  chaque page wiki du projet avec :
        - page_title : titre de la page wiki
        - file_name : nom du fichier
        - file_date : date de crÃ©ation du fichier
        - file_type : type gÃ©nÃ©ral (image, pdf, texte, etc.)
        - file_size : taille du fichier en octets
        - download_url : URL directe pour tÃ©lÃ©charger le fichier
        """
        project = self.redmine.project.get(project_identifier)
        wiki_pages = project.wiki_pages

        attachments_info = []

        for page in wiki_pages:
            full = self.redmine.wiki_page.get(resource_id=page.title, project_id=project.identifier, include='attachments')
            if hasattr(full, 'attachments'):
                for att in full.attachments:                    
                    file_extension = att.filename.split('.')[-1] if '.' in att.filename else 'unknown'
                    attachments_info.append({
                        'page_title': full.title,
                        'file_name': att.filename,
                        'file_date': att.created_on.strftime("%Y-%m-%d %H:%M"),
                        'file_type': file_extension,
                        'file_size':att.filesize,
                        'file_category': self.categorize_filename(att.filename),
                        'download_url': att.content_url
                    })

        return attachments_info


    def publish_project_wiki_table(self,project_identifier: str,
                        publish_page: str, wiki_data: pd.DataFrame, comment: str = None):
        """
        Publish a wiki page containing a table of the provided wiki_data.

        wiki_data: list of dicts as returned by scrap_wiki_pages
        publish_page: title for the new wiki page (identifier)
        comment: optional update comment
        """
        #project = self.redmine.project.get(project_identifier)
        parent_page = self.input_rag_wiki_pages_parent

        lines = []
        header = ["Project name", "Project id", "Page title", "Link","Parent page", "Hierarchy", "Creation", "Latest update", "Category","Status"]
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
                row['wiki_category'],
                row['status']
            ]
            lines.append("| " + " | ".join(cells) + " |")
        text = "\n".join(lines)

        try:
            wiki_page = self.redmine.wiki_page.get(resource_id=publish_page, project_id=self.publish_project)
            wiki_page.text = text
            wiki_page.comments = comment or f"Updated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            wiki_page.save()
            print(f"Page '{publish_page}' updated")
        except:
            self.redmine.wiki_page.create(
                project_id=self.publish_project,
                title=publish_page,
                parent_title=parent_page,
                text=text,
                comments=comment or f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"Page '{publish_page}' created.")




    def publish_wiki_attachements_table(self,project_identifier: str,
                                        publish_page: str, wiki_data: pd.DataFrame, comment: str = None):
        """
        Publish a Redmine wiki page containing a table with one row per attachment.

        :param redmine_url: Redmine instance URL
        :param api_key: API key for authentication
        :param project_identifier: Redmine project identifier
        :param target_page: Title of the wiki page to create or update
        :param wiki_data: pandas.DataFrame with flattened attachment info, one row per attachment
        :param comment: Optional update comment
        """
        project = self.redmine.project.get(project_identifier)
        parent_page = self.input_rag_attached_files_parent


        lines = []
        header = ["Project", "Wiki Page", "Parent", "File Name", "Date", "Type", "Size (KB)", "Download", "Category","Status"]
        lines.append("| " + " |".join(f"*{h}*" for h in header) + " |")
        
        df1 = wiki_data.explode("attachments")
        df1 = df1[df1["attachments"].notnull()]

        df2 = pd.json_normalize(df1["attachments"])
        df3 = pd.concat([df1.drop(columns=["attachments", "page_title"]).reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
        
        for _, row in df3.iterrows():
            size_kb = f"{int(row.get('file_size', 0)) // 1024}" if pd.notnull(row.get('file_size')) else ''
            download_link = f"{row.get('download_url')}" if pd.notnull(row.get('download_url')) else ''
            print(row)
            cells = [
                row.get('project_name', ''),
                row.get('page_title', ''),
                row.get('parent', ''),
                row.get('file_name', ''),
                row.get('file_date', '')[:10],
                row.get('file_type', ''),
                size_kb,
                download_link,
                row.get('file_category', ''),
                row.get('status', '')
            ]
            
            # Force conversion en string
            cells = [str(cell) for cell in cells]
            lines.append("| " + " |".join(cells) + " |")


        text = "\n".join(lines)

        try:
            wiki_page = self.redmine.wiki_page.get(resource_id=publish_page, project_id=self.publish_project)
            wiki_page.text = text
            wiki_page.comments = comment or f"Updated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            wiki_page.save()
            print(f"Wiki page '{publish_page}' updated.")
        except:
            self.redmine.wiki_page.create(
                project_id=self.publish_project,
                title=publish_page,
                parent_title=parent_page,
                text=text,
                comments=comment or f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            print(f"Wiki page '{publish_page}' created.")


    def extract_wiki_table_and_content_to_json(self, project_identifier: str, table_page: str):
        """
        Read a Redmine wiki page containing a metadata table and fetch corresponding wiki pages content.
        Save as JSON with content + metadata per page for RAG ingestion.

        The table must contain columns:
        *Projet* |*Title* |*Link* |*Parent page* |*Hierarchy* |*Creation* |*Latest update* |*Status*
        """

        table_page_data = self.redmine.wiki_page.get(table_page,project_id=self.publish_project)
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
                    wiki_page = self.redmine.wiki_page.get(wiki_page_name, project_id=project_identifier)
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
                        "category": row.get("category", ""),
                        "status": row.get("status", "")
                    }
                }
                documents.append(doc)
                
        project_name = row.get("project_name", "")
        output_json_file = f"{self.output_wiki_json_filename}_{project_name}.json"
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(documents)} entries to {output_json_file}")
        
        return output_json_file



if __name__ == "__main__":  
            
    redminedb = RedmineScraper(some_projects_1, publish_project="ops")
    redminedb.scrap_and_publish_redmine_projects()
