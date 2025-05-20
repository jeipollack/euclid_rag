#!/usr/bin/env python
import csv
from get_data import get_data_section_full
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Hard-coded inputs
TOPICS_CSV = 'topics2get.csv'
TOPICS_NUMBER_LIMIT = 3  # 0 = no limit

def main():
    # Iterate topics and fetch full allowed sections via helper
    texts, metadatas = [], []
    with open(TOPICS_CSV, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for i, (name, url) in enumerate(reader):
            if TOPICS_NUMBER_LIMIT > 0 and i >= TOPICS_NUMBER_LIMIT:
                break
            results = get_data_section_full(url, name)
            for item in results:
                # print(item['content'])
                # print({k: v for k, v in item.items() if k != 'content'})
                # print('-' * 120)
                texts.append(item['content'])
                metadatas.append({k: v for k, v in item.items() if k != 'content'})
    # 3) embed & persist FAISS vector store
    embed = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-1.7B")
    # embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts=texts, embedding=embed, metadatas=metadatas)
    db.save_local("faiss_index")
    print("✅ Persisted FAISS index to faiss_index/")

if __name__ == "__main__":
    main()