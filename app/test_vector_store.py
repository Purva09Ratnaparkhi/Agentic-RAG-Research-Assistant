# app/test_vector_store.py

from app.ingestion import ingest_document
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore

DOCUMENT_PATH = "data/documents/sample.pdf"

def main():
    print("Ingesting document...")
    chunks = ingest_document(DOCUMENT_PATH)

    print("Loading embedding model...")
    embedder = EmbeddingModel()

    print("Embedding chunks...")
    embeddings = embedder.embed_documents(chunks)

    print("Initializing vector store...")
    store = VectorStore()

    print("Storing documents in Chroma...")
    store.add_documents(chunks, embeddings)

    print("Ready! Now testing search...\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        q_emb = embedder.embed_query(query)
        results = store.search(q_emb, k=3)

        print("\nTop results:\n")
        for i, doc in enumerate(results["documents"][0]):
            print(f"Result {i+1}:\n{doc[:500]}\n")


if __name__ == "__main__":
    main()
