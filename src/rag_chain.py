from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# ---- This file connects everything together ----
# PDF chunks in ChromaDB → retrieve relevant ones
# Pass them as context to LLM
# LLM answers using only that context
# Remember conversation history

def load_vectorstore(db_path: str = "./chroma_db") -> Chroma:
    """
    Loads existing ChromaDB from disk.
    We don't recreate it — just load what was already indexed.
    This makes second run much faster.
    """
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    return vectorstore

def get_retriever(vectorstore: Chroma):
    """
    Creates retriever from vectorstore.
    k=5 means return top 5 most relevant chunks.
    More chunks = better context but slower response.
    """
    return vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

def format_docs(docs) -> str:
    """
    Joins retrieved chunks into one string.
    Also adds chunk numbers so AI can cite sources.
    """
    formatted = []
    for i, doc in enumerate(docs):
        formatted.append(f"[Chunk {i+1}]:\n{doc.page_content}")
    return "\n\n".join(formatted)

def create_rag_chain(retriever):
    """
    Creates the full RAG chain with conversation memory.
    
    MessagesPlaceholder → injects full chat history
    context → relevant chunks from your PDF
    question → current user question
    """
    llm = OllamaLLM(model="llama3.2")
    parser = StrOutputParser()
    
    # System prompt tells AI to:
    # 1. Answer ONLY from context
    # 2. Cite which chunk answered
    # 3. Say "not found" if answer isn't in document
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI Study Assistant helping students understand documents.

STRICT RULES:
1. Answer ONLY the current question
2. Use ONLY the context provided below
3. Never repeat previous questions or answers
4. Keep answer concise — maximum 3 sentences
5. If answer found, mention chunk number
6. If not found, say: 'This information is not in the document.'

Context from document:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
    
    # chain = prompt → llm → parser
    chain = prompt | llm | parser
    return chain

class StudyAssistant:
    """
    Main class that combines everything:
    - PDF retrieval
    - Conversation memory
    - RAG answering
    
    Using a class keeps all state (history, retriever)
    in one place — clean and organised.
    """
    
    def __init__(self, db_path: str = "./chroma_db"):
        print("Loading knowledge base...")
        # Load existing ChromaDB — no re-embedding needed
        vectorstore = load_vectorstore(db_path)
        self.retriever = get_retriever(vectorstore)
        self.chain = create_rag_chain(self.retriever)
        
        # chat_history stores full conversation
        # grows with every turn — same as memory_chat.py
        self.chat_history = []
        print("Study Assistant ready!\n")
    
    def ask(self, question: str) -> dict:
        """
        Takes a question, retrieves context, generates answer.
        Returns answer + source chunks for citation.
        """
        # Step 1 — find relevant chunks
        relevant_docs = self.retriever.invoke(question)
        context = format_docs(relevant_docs)
        
        # Step 2 — generate answer with full history
        answer = self.chain.invoke({
            "context": context,
            "history": self.chat_history,
            "question": question
        })
        
        # Step 3 — save to memory
        # Both question and answer added to history
        self.chat_history.append(
            HumanMessage(content=question)
        )
        self.chat_history.append(
            AIMessage(content=answer)
        )
        
        return {
            "answer": answer,
            "sources": [doc.page_content[:150] for doc in relevant_docs],
            "chunks_used": len(relevant_docs)
        }
    
    def clear_history(self):
        """Resets conversation — fresh start."""
        self.chat_history = []
        print("Conversation history cleared.")