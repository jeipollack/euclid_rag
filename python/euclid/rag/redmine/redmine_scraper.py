
# Copyright (C) 2025 Euclid Science Ground Segment
# Licensed under the GNU LGPL v3.0.
# See <https://www.gnu.org/licenses/>.

"""
RedmineScraper provides methods to interact with a Redmine instance for the purpose of extracting,
processing, and publishing project and wiki page metadata, as well as attachments, to facilitate
downstream tasks such as RAG (Retrieval-Augmented Generation) ingestion for chatbots.
This class supports:
- Listing all active Redmine projects and exporting them to a wiki page as a table.
- Selecting projects for further processing based on a selection mode.
- Scraping wiki pages and their metadata for selected projects.
- Publishing tables of wiki pages and attachments to Redmine wiki pages.
- Extracting wiki page content and metadata to JSON files for RAG ingestion.
- Concatenating multiple JSON files into a single file for downstream use.
- Designed for use in multi-Step CLI pipelines for Redmine data extraction and publication.


Environment Variables
--------------------
MY_REDMINE_TOKEN : str
    The API token for authenticating with the Redmine instance.
MY_REDMINE_URL : str
    The base URL of the Redmine instance.
"""
from redminelib import Redmine
from redminelib.exceptions import ResourceNotFoundError
from datetime import datetime
import os
import pandas as pd
import json
import argparse
from log_config import setup_logger

logger = setup_logger(__name__,log_file="redmine_scraper.log")

class RedmineScraper:
    def __init__(self,
                 publish_project:str,
                 data_folder:str="./data",
                 get_wiki_pages:bool=True,
                 get_attachments:bool=False):
        """
        Initialize the RedmineScraper instance.
        Parameters
        ----------
        publish_project : str
            The name of the project to publish or process.
        data_folder : str, optional
            The folder where data files will be stored (default is "./data").
        get_wiki_pages : bool, optional
            Whether to retrieve wiki pages from the Redmine project (default is True).
        get_attachments : bool, optional
            Whether to retrieve attachments from the Redmine project (default is False).
        Notes
        -----
        - Requires the environment variables 'MY_REDMINE_TOKEN' and 'MY_REDMINE_URL' to be set.
        - Initializes Redmine API connection and sets up data directories and filenames for output.
        """
        
        api_key = os.environ.get('MY_REDMINE_TOKEN')
        self.redmine_url = os.environ.get('MY_REDMINE_URL')
        self.redmine = Redmine(self.redmine_url, key=api_key)
        
        self.get_wiki_pages = get_wiki_pages
        self.get_attachments = get_attachments
        self.extract_content_to_json = True
        
        
        #self.redmine_projects_to_browse = redmine_projects_to_browse
        self.publish_project = publish_project
        self.input_rag_wiki_pages_parent= "Rag_wiki_pages"
        self.input_rag_attached_files_parent= "Rag_attached_files"
        self.output_attached_json_filename="rag_attached"
        self.output_wiki_json_filename="rag_wiki"
        
        self.selected_redmine_projects_wiki_page = "Selected_Redmine_Projects"
        self.all_redmine_projects_wiki_page = "All_Redmine_Projects"

        self.data_folder = data_folder
        self.selected_redmine_projects_json_file = None
        
        os.makedirs(self.data_folder, exist_ok=True)

       

    def init_list_of_redmine_projects(self):
        """ 
        Export all active Redmine projects to a wiki page table.
        
        This method retrieves all active (non-archived) Redmine projects, compiles their details
        (including project name, hierarchy path, number of wiki pages, and URL), and exports them
        as a markdown table to a specified wiki page. The table includes a 'Selection' column for
        further categorization.
        Parameters
        ----------
        Notes
        -----
        - Only projects with status == 1 (active) are included.
        - The wiki page is created if it does not exist.
        - The table includes columns for project index, name, path, number of wiki pages, URL, and selection status.
        - Errors retrieving wiki pages for a project are logged.
        """
        
        def truncate(text, width):
            """Truncate text to fit within the given width."""
            return text if len(text) <= width else text[:width-3] + '...'

        wiki_page  = self.selected_redmine_projects_wiki_page
        wiki_project = self.publish_project
        
        projects = self.redmine.project.all()
        
        projects_with_path = self.add_hierarchy_path_to_projects(projects)
                
        lines = [f"\nPage created on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        lines.append("h2. Redmine Project list")
        lines.append("\nEdit the 'Selection' column to select the projects to be scraped for the Euclid chatbot.\n\n")
        lines.append("\nPossible values are: 'Publication', 'Internal Major', 'Internal Minor','None'\n")


        # Publication \n Internal Major \n Internal minor \n TBD
        lines.append("{font-weight:bold; background:lightgreen}. |_. # |_. Project Name  |_. Project Path |_. #Wiki pages |_. URL |_. Selection |")
        logger.info(". |_. # |_. Project Name  |_. Project Path |_. #Wiki pages |")
        
        logger.info(f"{'#':<5} | {'Project Name':<10} | {'Project Path':<20} | {'#Wiki pages':<30}")
        logger.info("-" * 80)


        for idx, project in enumerate(projects, 1):
            if project.status != 1:  # Only include active projects
                continue

            name = project.name            
            
            # TODO: fix the description in order to not have textile characters
            #description = getattr(project, 'description', '').replace('\n', ' ') if hasattr(project, 'description') else ''
            
            wiki_count = 0
            try:
                wiki_pages = project.wiki_pages
                wiki_count = len(list(wiki_pages))
            except Exception as e:
                logger.error(f"Error retrieving wiki pages for {project.name}: {e}")
                            
            
            #parent = project.parent.name if hasattr(project, 'parent') else ''
            project_path = projects_with_path[name]
            url = f'"{project.identifier}":{self.redmine_url}/projects/{project.identifier}'
            status = "None"

            #line = f"| {name} | {wiki_count} | {parent} | {url} | {status} |"
            line = f"| {idx} | {name}  | {project_path} |  {wiki_count} | {url} | {status} |"
            logger.info(f"| {idx:<5} | {truncate(name, 30):<30} | {truncate(project_path, 50) :<50} | {wiki_count:<3} |")
    
            lines.append(line)

        textile_table = "\n".join(lines)

        # Try to update the wiki page, or create it if it doesn't exist
        try:
            page = self.redmine.wiki_page.get(wiki_page, project_id=wiki_project)
            page.text = textile_table
            page.save()
        except:
            self.redmine.wiki_page.create(project_id=wiki_project, title=wiki_page, text=textile_table)
            
        logger.info(f"Wiki page '{wiki_page}' created in project '{wiki_project}'.")


        def get_wiki_page(self, project_identifier: str, page_title: str):
            """
            Retrieve a specific wiki page from a Redmine project.

            Parameters
            ----------
            project_identifier : str
                The unique identifier of the Redmine project.
            page_title : str
                The title of the wiki page to retrieve.

            Returns
            -------
            wiki_page : redminelib.resources.WikiPage
                The requested wiki page object.

            Raises
            ------
            redminelib.exceptions.ResourceNotFoundError
                If the project or wiki page does not exist.
            redminelib.exceptions.AuthError
                If authentication fails.
            """
            project = self.redmine.project.get(project_identifier)
            wiki_page = self.redmine.wiki_page.get(resource_id=page_title, project_id=project.identifier)
            return wiki_page
    
    
    def read_list_of_redmine_projects(self, selection_mode: str = "Publication"):
        """
        Read a Redmine wiki page containing a metadata table and filter projects by selection mode.

        Parameters
        ----------
        selection_mode : str, optional
            The selection mode to filter projects (default is "Publication").

        Returns
        -------
        None

        Side Effects
        ------------
        - Saves the filtered list of selected projects as a JSON file in the data folder.
        - Sets self.selected_redmine_projects_json_file to the output file path.

        Notes
        -----
        The table must contain columns:
        *Projet* |*Title* |*Link* |*Parent page* |*Hierarchy* |*Creation* |*Latest update* |*Status*
        """
        table_page = self.selected_redmine_projects_wiki_page

        table_page_data = self.redmine.wiki_page.get(table_page, project_id=self.publish_project)
        lines = table_page_data.text.splitlines()

        projects = []
        headers = []
        for line in lines:
            if "|_. # " in line.strip():
                headers = [h.strip("_._").lower().replace(" ", "_") for h in line.strip("|").split("|")]
                headers = [item.strip('_') for item in headers]
                headers = headers[1:]
            elif line.strip().startswith("|") and headers:
                cells = [c.strip() for c in line.strip("|").split("|")]
                row = dict(zip(headers, cells))
                projects.append(row)

        selected_projects = [project for project in projects if project.get("selection") == selection_mode]
        if not selected_projects:
            logger.warning(f"No projects found with selection mode '{selection_mode}'.")
            output_json_file = os.path.join(self.data_folder, f"{table_page}_empty.json")
        else:
            logger.info(f"Found {len(selected_projects)} projects with selection mode '{selection_mode}'.")
            output_json_file = os.path.join(self.data_folder, f"{table_page}.json")

        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(selected_projects, f, ensure_ascii=False, indent=2)

        logger.debug(f"Saved {len(projects)} projects to {output_json_file}")
        self.selected_redmine_projects_json_file = output_json_file
        
    
    
    def add_hierarchy_path_to_projects(self, projects):
        """
        Build a dictionary mapping project names to their full hierarchy paths.

        Parameters
        ----------
        projects : list
            List of Redmine project resource objects, each having at least
            'name' and 'parent' attributes (parent can be None or another project).

        Returns
        -------
        dict
            Dictionary keyed by project name, where each value is a string
            representing the full hierarchy path (e.g., "Parent > Child > Subchild").
        """
        # Build a map from project name to project object for quick parent lookup
        project_map = {p.name: p for p in projects}

        def get_path(project):
            path = []
            current = project
            while current:
                path.append(current.name)
                current = getattr(current, 'parent', None)
            return " > ".join(reversed(path))

        hierarchy_paths = {}
        for project in projects:
            hierarchy_paths[project.name] = get_path(project)

        return hierarchy_paths


    
    def get_wiki_attachments(self, project_identifier: str, page_title: str):
        """
        Retrieve the list of attachments for a specific wiki page in a Redmine project.

        Parameters
        ----------
        project_identifier : str
            The unique identifier of the Redmine project.
        page_title : str
            The title of the wiki page for which to retrieve attachments.

        Returns
        -------
        list
            A list of attachment objects associated with the specified wiki page.
            Returns an empty list if no attachments are found.

        Notes
        -----
        - Requires that the Redmine API connection is properly initialized.
        - Each attachment object contains metadata such as filename, filesize, content_url, etc.
        """
        project = self.redmine.project.get(project_identifier)
        wiki_page = self.redmine.wiki_page.get(resource_id=page_title, project_id=project.identifier, include='attachments')
        attachments = wiki_page.attachments if hasattr(wiki_page, 'attachments') else []
        return attachments
    
    def scrap_and_publish_redmine_projects(self,read_only:bool=False):
        """ Scrape and publish wiki pages and attachments for selected Redmine projects.
        Parameters
        ----------
        read_only : bool, optional
            If True, only returns the names of the wiki pages that would be published,
            without performing any scraping or publishing actions. Default is False.
        Returns
        -------
        published_pages : list of str
            A list of names of the published (or to-be-published, if read_only=True)
            wiki pages for each selected Redmine project.
        Notes
        -----
        - Reads the list of selected Redmine projects from a JSON file.
        - For each project, scrapes the wiki pages and optionally attachments.
        - Publishes the scraped data as wiki page tables unless `read_only` is True.
        - If no wiki pages are found for a project, publication is skipped for that project.
        - If `self.get_attachments` is True, also scrapes and publishes attachments for each project.
        """
   
        with open(self.selected_redmine_projects_json_file, "r", encoding="utf-8") as f:
            selected_projects = json.load(f)  
            
        logger.debug(f"Total number of Redmine projects to browse: {len(selected_projects)}")
        # Filter projects based on selection mode
                        
        # Loop through each project and scrape wiki pages   
        
        published_pages = []
        
    
        for project_dict in selected_projects:
            
            project_name = project_dict.get("project_name")
            url = project_dict.get("url")
            project_path = project_dict.get("project_path")
            project = url.rstrip('/').split('/')[-1] 

            logger.debug(f"Scraping project: {project_name} ({project})")
            publish_page = f"Wiki_pages_from_{project}"
            
            # Check if the project is already published. If yes, skip scraping
            # and return the wiki pages that list all the wiki pages of the Redmine project
            if read_only:
                published_pages.append(publish_page)
                continue

            
            df_wiki = self.scrap_project_wiki_pages(project)
            logger.info(f"Scraped {len(df_wiki)} wiki pages for project {project}")
            # Check if the DataFrame is empty
            if df_wiki.empty:
                logger.warning(f"No wiki pages found for project {project}. Skipping publication.")
                continue
            self.publish_project_wiki_table(project_path, publish_page, df_wiki)
            logger.debug(f"Published wiki page table for project {project} to {publish_page}")
            
            published_pages.append(publish_page)
            
                
            if self.get_attachments:
                logger.info(f"Scraping attachments for project: {project}")
                publish_page = f"Attachments_from_{project}"
                self.publish_wiki_attachements_table(project, project_path,publish_page, df_wiki)
                logger.info(f"Published wiki attachments table for project {project} to {publish_page}")
                
        return published_pages
                


    def scrap_project_wiki_pages(self,project_identifier: str) -> pd.DataFrame:
        """
        Scrape all wiki pages for a given Redmine project and return their metadata as a pandas DataFrame.
        Parameters
        ----------
        project_identifier : str
            The identifier of the Redmine project whose wiki pages are to be scraped.
        Returns
        -------
        pd.DataFrame
            A DataFrame containing metadata for each wiki page, with columns:
            - project_name: str
                The name of the project.
            - project_identifier: str
                The identifier of the project.
            - page_title: str
                The title of the wiki page.
            - parent: str
                The title of the parent wiki page, if any.
            - full_hierarchy: str
                The full hierarchy path of the wiki page.
            - status: str
                The publication status of the wiki page ("Publication" or "None").
            - created_on: str
                The creation timestamp of the wiki page (formatted as "YYYY-MM-DD HH:MM").
            - updated_on: str
                The last updated timestamp of the wiki page (formatted as "YYYY-MM-DD HH:MM").
            - wiki_category: str
                The category assigned to the wiki page based on its title.
            - attachments: list
                List of attachments associated with the wiki page.
        Notes
        -----
        - If no wiki pages are found for the project, an empty DataFrame is returned.
        - The status is set to "None" if the page title contains keywords such as "old", "draft", "tmp", or "archive".
        """
        project = self.redmine.project.get(project_identifier)
        wiki_pages = project.wiki_pages

        data = []
        logger.info(f"Scraping wiki pages for project: {project.name} ({project.identifier})")
        if not wiki_pages:
            logger.warning(f"No wiki pages found for project {project.name} ({project.identifier}).")
            return pd.DataFrame()
        logger.info(f"Found {len(wiki_pages)} wiki pages.")
        # Loop through each wiki page and extract metadata
        
        logger.info("-" * 100)
        logger.info(f"{'Project Name':<40} | {'Wiki Path':<40} | {'updated_on':<20}")
        logger.info("-" * 100)
                
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
            
            
            updated_on = full.updated_on.strftime("%Y-%m-%d %H:%M") if hasattr(full, 'updated_on') else ''
            hierarchy = "/".join(hierarchy)
            logger.info(f"{project.name:<40} | {hierarchy:<40} | {updated_on:<20}")


            if self.get_attachments:
                logger.debug ("\nWiki page attachments:\n")
                attachments_list = self.get_wiki_attachments(project_identifier,page)
                for att in attachments_list:
                    logger.debug(att)
                    
            # Set status to "None" if the title contains certain keywords
            if any(keyword in full.title.lower() for keyword in ["old", "draft", "tmp", "archive"]):
                status = 'None'
            else:
                status = 'Publication'
                
            data.append({
                'project_name': project.name,
                'project_identifier': project.identifier,
                'page_title': full.title,
                'parent': full.parent.title if hasattr(full, 'parent') else '',
                'full_hierarchy': hierarchy,
                'status': status,
                'created_on': full.created_on.strftime("%Y-%m-%d %H:%M") if hasattr(full, 'created_on') else '',
                'updated_on': updated_on,
                'wiki_category': self.categorize_filename(full.title),
                'attachments': attachments_list
            })

        df_wiki = pd.DataFrame.from_dict(data)

        return df_wiki


    def categorize_filename(sel,filename: str) -> str:
        """
        Categorizes a filename based on predefined keyword groups.
        The function checks if the filename contains any of the keywords associated with specific categories
        such as meeting minutes, project documents, or plans. If a match is found, the corresponding category
        is returned. If no keywords match, "default" is returned.
        Parameters
        ----------
        sel : Any
            Placeholder for the instance or class reference (not used in this function).
        filename : str
            The name of the file to categorize.
        Returns
        -------
        str
            The category of the file. Possible values are:
            - "meeting_minutes"
            - "pf_document"
            - "plan"
            - "default"
        """
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
        Retrieve a list of attachments for each wiki page in a given Redmine project.
        Parameters
        ----------
        project_identifier : str
            The identifier of the Redmine project.
        page : 
            Unused parameter (reserved for future use or compatibility).
        Returns
        -------
        list of dict
            A list where each element is a dictionary containing information about an attachment:
                - 'page_title': str
                    Title of the wiki page.
                - 'file_name': str
                    Name of the attached file.
                - 'file_date': str
                    Creation date of the file (formatted as "YYYY-MM-DD HH:MM").
                - 'file_type': str
                    General type of the file (e.g., image, pdf, text, etc.), based on file extension.
                - 'file_size': int
                    Size of the file in bytes.
                - 'file_category': str
                    Category of the file as determined by `categorize_filename`.
                - 'download_url': str
                    Direct URL to download the file.
        Notes
        -----
        This method requires access to the Redmine API and assumes that the `self.redmine` client is properly configured.
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


    def publish_project_wiki_table(self,hierarchy_path:str,
                        publish_page: str, wiki_data: pd.DataFrame, comment: str = None):
        
        """
        Publish a Redmine wiki page containing a table with one row per wiki page.

        Parameters
        ----------
        hierarchy_path : str
            The hierarchy path of the project to be included in the wiki page.
        publish_page : str
            The title of the wiki page to create or update.
        wiki_data : pandas.DataFrame
            DataFrame with flattened wiki page info, one row per wiki page.
        comment : str, optional
            Optional update comment.
        """

        #project = self.redmine.project.get(project_identifier)
        parent_page = self.input_rag_wiki_pages_parent
        
        
        lines = [f"\nPage created on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        lines.append("h2. Redmine Wiki pages list")
        lines.append("\nEdit the 'Selection' column to select the wiki pages to be scraped for the Euclid chatbot.\n\n")
        lines.append("\nPossible values are: 'Publication', 'Internal Major', 'Internal Minor','None'\n")

        
        header = ["Project name", "Project id", "project path", "Wiki path", "Creation", "Latest update", "Category","Selection"]
        lines.append("{font-weight:bold; background:orange}. | " + " |".join(f"*{h}*" for h in header) + " |")
        for _,row in wiki_data.iterrows():
            cells = [
                row['project_name'],
                row['project_identifier'],
                hierarchy_path,
                #row['page_title'],
                #"[["+row['project_name']+":"+ row['page_title']+"|"+row['full_hierarchy']+"]]",
                "[["+row['project_name']+":"+ row['page_title']+"]]",
                #"[["+row['full_hierarchy']+":"+ row['page_title']+"]]",
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
            logger.info(f"Page '{publish_page}' updated")
        except:
            self.redmine.wiki_page.create(
                project_id=self.publish_project,
                title=publish_page,
                parent_title=parent_page,
                text=text,
                comments=comment or f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            logger.info(f"Page '{publish_page}' created.")




    def publish_wiki_attachements_table(self,project_identifier: str,
                                        publish_page: str, wiki_data: pd.DataFrame, comment: str = None):
        """
        Publish a Redmine wiki page containing a table with one row per attachment.

        Parameters
        ----------
        project_identifier : str
            The identifier of the Redmine project.
        publish_page : str
            The title of the wiki page to create or update.
        wiki_data : pandas.DataFrame
            DataFrame with flattened attachment info, one row per attachment.
        comment : str, optional
            Optional update comment.
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
            logger.info(row)
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
            logger.info(f"Wiki page '{publish_page}' updated.")
        except:
            self.redmine.wiki_page.create(
                project_id=self.publish_project,
                title=publish_page,
                parent_title=parent_page,
                text=text,
                comments=comment or f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            logger.info(f"Wiki page '{publish_page}' created.")
            
            
    

    def extract_wiki_table_and_content_to_json(self, table_page: str,selection_mode: str = "Publication"):
        """
        Read a Redmine wiki page containing a metadata table and fetch corresponding wiki pages content.
        Save as JSON with content + metadata per page for RAG ingestion.

        The table must contain columns:
        *Projet* |*Title* |*Link* |*Parent page* |*Hierarchy* |*Creation* |*Latest update* |*Status*

        Parameters
        ----------
        table_page : str
            The title of the Redmine wiki page containing the metadata table.
        selection_mode : str, optional
            The selection mode to filter wiki pages (default is "Publication").

        Returns
        -------
        None if the input wiki table page does not exist else...
        output_json_file : str
            Path to the output JSON file containing the extracted content and metadata.
        """

        try:
            table_page_data = self.redmine.wiki_page.get(table_page, project_id=self.publish_project)
        except ResourceNotFoundError:
            # case where a redmine project does not contain any (selected) wiki page
            logger.warning(f"The wiki page '{table_page}' does not exist in '{self.publish_project}'")
            return None  


        lines = table_page_data.text.splitlines()

        documents = []
        headers = []
        
        logger.info("-" * 100)
        logger.info(f"{'Project Name':<40} | {'Wiki Page name':<40} | {'updated_on':<20}")
        logger.info("-" * 100)

        for line in lines:
            if "| *" in line:
                headers = [h.strip("* ").lower().replace(" ", "_") for h in line.strip("|").split("|")]
                headers = headers[1:]
            elif line.strip().startswith("|") and headers:
                cells = [c.strip() for c in line.strip("|").split("|")]
                row = dict(zip(headers, cells))
                wiki_path = row.get("wiki_path", "").strip()
                wiki_page_name = wiki_path.split(':')[1].rstrip(']]').strip()
                
                selection = row.get("selection", "")
                if selection.lower() != selection_mode.lower():
                    logger.info(f"Skipping page '{wiki_page_name}' with selection '{selection}'")
                    continue
                
                # Get the project identifier from the row, default to empty string if not found
                # This assumes the project identifier is stored in the 'project_id' column
                project_identifier = row.get("project_id", "").strip()
                project_name = row.get("project_name", "")
                updated_on = row.get("latest_update", "")
                logger.info(f"{project_name:<40} | {wiki_page_name:<40} | {updated_on:<20}")


                try:
                    wiki_page = self.redmine.wiki_page.get(wiki_page_name, project_id=project_identifier)
                    content = wiki_page.text
                except Exception as e:
                    content = f"[ERROR: Could not load page '{wiki_page_name}': {e}]"

                doc = {
                    "content": content,
                    "metadata": {
                        "project": project_name,
                        "page_name": wiki_page_name,
                        "project_path": row.get("project_path", ""),
                        "project_id": project_identifier,
                        "wiki_path": wiki_path.strip('[] '),
                        "created_on": row.get("creation", ""),
                        "updated_on": updated_on,
                        "category": row.get("category", ""),
                        "status": selection
                    }
                }
                documents.append(doc)
                
        project_name = row.get("project_name", "")
        output_json_file = os.path.join(self.data_folder, f"{self.output_wiki_json_filename}_{project_name}.json")

        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(documents)} entries to {output_json_file}")
        
        return output_json_file

    def read_contents_of_selected_wiki_pages(self, published_pages: list,selection_mode: str = "Publication"):
        """
        Read the contents of selected wiki pages and extract their tables and content to JSON files.

        Parameters
        ----------
        published_pages : list of str
            List of wiki page names to process.
        selection_mode : str, optional
            The selection mode to filter wiki pages (default is "Publication").

        Returns
        -------
        list of str
            List of paths to the generated JSON files.
        """
        if not published_pages:
            logger.warning("No published pages provided. Returning empty list.")
            return []
        
        rag_json_files = []
            
        logger.info(f"Extracting wiki table and content to JSON for project")
        for publish_page in published_pages:
            output_json_path = self.extract_wiki_table_and_content_to_json(publish_page,selection_mode)
            if output_json_path:
                logger.debug(f"Extracted wiki table and content to JSON for project {publish_page} to {output_json_path}")        
                rag_json_files.append(output_json_path)
            
        logger.info(f"{len(rag_json_files)} Json files have been extracted from the selected {len(published_pages)} wiki pages.")
        return rag_json_files
    
    
    def concatenate_rag_files(self,rag_json_files: list):
        """
        Concatenate all JSON files into one using numpy-style docstring.

        Parameters
        ----------
        rag_json_files : list of str
            List of paths to JSON files to concatenate.

        Returns
        -------
        str
            Path to the concatenated output JSON file.
        """
        # Check if the list is empty 
        all_json_files = []     
        for json_file in rag_json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_json_files.extend(data) 
                
        # Save the concatenated data to a new JSON file
        output_json_file = os.path.join(self.data_folder, f"{self.output_wiki_json_filename}.json")       
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(all_json_files, f, ensure_ascii=False, indent=2)
                
        logger.info(f"Saved {len(all_json_files)} entries to {output_json_file}")
        return output_json_file

def parse_arguments():
    """
    Parse command-line arguments for managing Redmine projects and reading wiki pages.

    Available arguments:
    --init-projects    Initialize a list of Redmine projects.
    --read-wiki        Read wiki pages (no project name required).

    Returns:
        argparse.Namespace: An object containing the values of the provided arguments.
    """
    parser = argparse.ArgumentParser(
        description="CLI tool for initializing projects and reading Redmine wiki pages."
    )
    
    parser.add_argument("--publication_redmine_project_name", type=str, default="ops",
                        help="name of the Euclid SGS Redmine project in which the Redmine projects and wiki pages are printed for selection")
    
    
    parser.add_argument("--data_path", type=str, default="./data",
        help="parent local folder where the ouput data will be saved")

    parser.add_argument(
        '--init_projects',
        action='store_true',
        help='Step 1: Initialize a list of Redmine projects.'
    )

    parser.add_argument(
        '--init_wiki_pages',
        action='store_true',
        help='Step 2: Initialize a list of wiki pages based on the selected Redmine projects.'
    )
    
    parser.add_argument(
        '--read_wiki_pages',
        action='store_true',
        help='Step 3: Access to the selected wiki pages and extract their content to JSON files, one per redmine project.'
    )

    parser.add_argument(
        '--concat_json_rag_files',
        action='store_true',
        help='Step 3 (option): The contents of the extracted Json files are concatenated into one file needed by the chatbot'
    )

    return parser.parse_args()

def main():
    """
    Main function that dispatches functionality based on command-line arguments.
    """
    logger.debug("Starting Redmine Scraper")
    # Parse command-line arguments

    args = parse_arguments()
    
    scraper = RedmineScraper(publish_project=args.publication_redmine_project_name, data_folder=args.data_path)
    logger.info(f"Data will be saved in: {scraper.data_folder}")

    if args.init_projects:
        logger.info("Step 1: Initializing a list of all the Redmine projects and publishing it into a Redmine wiki page")
        # Initialize a list of Redmine projects and publish it to a main wiki page
        scraper.init_list_of_redmine_projects()


    if args.init_wiki_pages:
        logger.info("Step 2: Initializing a list of Redmine wiki pages...")
        # Initialize a list of Redmine wiki pages per selected Redmine project () and publish it to a wiki page
        scraper.read_list_of_redmine_projects(selection_mode="Publication")
        logger.info(f"Selected Redmine projects file: {scraper.selected_redmine_projects_json_file}")
        published_pages_created = scraper.scrap_and_publish_redmine_projects()
        logger.info(f"Published wiki pages: {published_pages_created}")        
        
    if args.read_wiki_pages:
        logger.info("Step 3: Reading the Redmine wiki pages...")
        scraper.read_list_of_redmine_projects(selection_mode="Publication")
        existing_published_pages = scraper.scrap_and_publish_redmine_projects(read_only=True)
        if not existing_published_pages:
            logger.warning("No published pages found. Exiting.")
            return
        
        logger.debug(f"Published wiki pages for reading: {existing_published_pages}")        
        rag_json_files = scraper.read_contents_of_selected_wiki_pages(existing_published_pages,selection_mode="Publication")
        logger.info(f"RAG JSON files created: {rag_json_files}")

        if args.concat_json_rag_files:
            logger.debug(f"Concatenating the extracted Json files: {existing_published_pages}")        
            output_final_tag_file = scraper.concatenate_rag_files(rag_json_files)
            logger.info(f"Concatenated RAG JSON file: {output_final_tag_file}")

        logger.info("You can now run the Ollama server with the following command:")
        logger.info("export KMP_DUPLICATE_LIB_OK=TRUE")
        logger.info("ollama serve --model granite3.1:2b --port 11434")
        logger.info("python check_redmine/mock_pipeline.py")
        
  
if __name__ == "__main__":
    main()