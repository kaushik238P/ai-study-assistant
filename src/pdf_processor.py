import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- How this file works ----
# 1. Read PDF or TXT → extract raw text
# 2. Split text into chunks
# 3. Convert chunks to vectors (embeddings)
# 4. Store vectors in ChromaDB

def read_pdf(file_path: str) -> str:
    """
    Reads a PDF or TXT file and returns all text as a string.
    - TXT files: read directly
    - PDF files: use PyMuPDF (fitz) to extract text
    """
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Read text file, {len(text)} characters")
        return text

    # PDF path — uses PyMuPDF
    import fitz
    doc = fitz.open(file_path)
    full_text = ""
    page_count = len(doc)

    for page_num, page in enumerate(doc):
        # get_text() extracts all readable text from one page
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    print(f"Read {page_count} pages, {len(full_text)} characters")
    return full_text


def split_text(text: str) -> list:
    """
    Splits large text into smaller overlapping chunks.

    chunk_size=300   → each chunk is ~300 characters
    chunk_overlap=100 → 100 chars shared between chunks
                        overlap prevents losing context
                        at chunk boundaries
    """
    if not text.strip():
        print("No text to split!")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
        # tries to split at paragraphs first,
        # then lines, then sentences, then words
    )
    chunks = splitter.create_documents([text])
    print(f"Created {len(chunks)} chunks")
    return chunks


def store_in_chromadb(chunks: list, db_path: str = "./chroma_db") -> Chroma:
    """
    Converts chunks to vectors and stores in ChromaDB.

    OllamaEmbeddings converts text → numbers (vectors)
    Chroma.from_documents stores those vectors on disk
    persist_directory saves DB so it survives restart
    """
    if not chunks:
        raise ValueError("No chunks to store! Check if your file has readable text.")

    # Clear old DB if exists — fresh start every upload
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print("Cleared old database")

    embeddings = OllamaEmbeddings(model="llama3.2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def process_pdf(file_path: str) -> Chroma:
    """
    Main function — runs all 3 steps together.
    Returns a vectorstore ready for querying.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\nProcessing: {file_path}")
    print("-" * 40)

    # Step 1 — Read
    text = read_pdf(file_path)

    # Step 2 — Split
    chunks = split_text(text)

    # Step 3 — Store
    vectorstore = store_in_chromadb(chunks)

    print("PDF processed successfully!")
    return vectorstore