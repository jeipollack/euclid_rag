# DPDD ingestion scripts

We put here some scripts to scrap DPDD data and ingest it into some VectorStore.

We still need a csv file **topic2get.csv** with the list of URLs that we want to proceed on DPDD site.

Inside of **dpdd_ingestion_config** directory we have dpdd_config.yaml file with some configs. For the moment there is a list of sections what we want to ingest into VectorStore. These section will be ommited during ingestion.

The main file is _populate_dpdd_vector_store.py_. It will populate FAISS indexes in **VectorStore_indexes**.
It's rewritten to follow the same structure (and logic) as ingestion scripts for pdfs.
It could be used with existing Vector Stores : normally it will just add data there.

But for the moment I still keep it here.
TODO : move this script to python/rag/extra_scripts

I add **requirements.txt** as well. Normally it should be pretty the same as for main project, but it allow to use these scripts as some autonomus project.
