from src.pdf_processor import process_pdf
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Test with any PDF you have
# Download a free one if you don't have one
# Try your college notes or any textbook chapter

pdf_path = "sample.txt"  # put any PDF here

# Process the PDF
vectorstore = process_pdf(pdf_path)

# Test retrieval
print("\nTesting retrieval...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

test_query = "what is this document about?"
docs = retriever.invoke(test_query)

print(f"\nQuery: {test_query}")
print(f"Found {len(docs)} relevant chunks:")
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:")
    print(doc.page_content[:200])