# app/ingestion.py

from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf


def load_document(path: str):
    """
    Parse a document into structured elements using unstructured.
    """
    elements = partition_pdf(filename=path)
    return elements


def elements_to_text_chunks(elements) -> List[str]:
    """
    Convert unstructured elements (text, tables, titles, etc.)
    into LLM-friendly text chunks.
    """
    chunks = []

    for el in elements:
        text = str(el).strip()
        if not text:
            continue

        # Light normalization
        text = " ".join(text.split())
        chunks.append(text)

    return chunks


def chunk_text(text_chunks: List[str], max_chars: int = 800, overlap: int = 100):
    """
    Merge small chunks and split large ones into size-bounded overlapping chunks.
    """
    final_chunks = []
    buffer = ""

    for piece in text_chunks:
        if len(buffer) + len(piece) < max_chars:
            buffer += " " + piece
        else:
            final_chunks.append(buffer.strip())
            buffer = piece

    if buffer.strip():
        final_chunks.append(buffer.strip())

    # Add overlap by re-splitting large chunks
    overlapped = []
    for chunk in final_chunks:
        if len(chunk) <= max_chars:
            overlapped.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                end = start + max_chars
                sub = chunk[start:end]
                overlapped.append(sub.strip())
                start = end - overlap

    return overlapped


def ingest_document(path: str):
    """
    Full ingestion pipeline:
    Document -> unstructured elements -> normalized text -> chunks
    """
    print("Parsing document with unstructured...")
    elements = load_document(path)

    print(f"Extracted {len(elements)} elements")

    print("Normalizing elements to text...")
    raw_chunks = elements_to_text_chunks(elements)

    print("Chunking...")
    chunks = chunk_text(raw_chunks)

    print(f"Final chunks: {len(chunks)}")

    return chunks


if __name__ == "__main__":
    sample_path = "data/documents/sample.pdf"

    if Path(sample_path).exists():
        chunks = ingest_document(sample_path)
        print("\n--- Preview first 5 chunks ---\n")
        for i, c in enumerate(chunks[:5]):
            print(f"Chunk {i+1}:\n{c[:500]}\n")
    else:
        print("Put a document in data/documents/sample.pdf to test.")
