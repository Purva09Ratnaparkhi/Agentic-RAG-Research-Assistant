# app/vector_store.py

import chromadb
from chromadb.config import Settings
from typing import List


class VectorStore:
    def __init__(self, persist_dir: str = "data/chroma"):
        self.client = chromadb.Client(
            Settings(persist_directory=persist_dir, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name="documents")

    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
        )

        # # Persist to disk
        # self.client.persist()

    def search(self, query_embedding: List[float], k: int = 5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        return results
