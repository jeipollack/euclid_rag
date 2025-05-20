# DPDD ingestion scripts

We put here some scripts to scrap DPDD data and ingest it into some VectorStore.

There are 3 csv files here :
* **topic2get.csv** : list of first level topics with links to get information from (it makes first content filtering) ;
* **h1_sections.csv** and **other_sections.csv** : list of second level sections that we shoudl/could use for our RAG.

Actually we get all H1 level sections.  
And do some arbitrary filtering of for other sections (H2, H3). 

Obvious these csv files were not created manually, but it was some handy-dirty scripting so I don't put such scripts here.

So two python script here are : get_data.py and populate_vector_store.py

- _get_data.py_ : has two function to parse DDPD pages and scrap the data.
- _populate_vector_store.py_ : call function from get_data and should populate some vector store.

  I add **requirements.txt** as well. Normally it should be pretty the same as for main project, but it allow to use these scripts as some autonomus project.
