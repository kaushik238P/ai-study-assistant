from src.pdf_processor import process_pdf
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---- How this file works ----
# 1. Load the vectorstore (ChromaDB)
# 2. Build a RAG chain (retrieve + generate)
# 3. Test with drone-related questions
# 4. Test conversation memory

class StudyAssistant:
    """
    RAG-based study assistant with conversation memory.
    
    RAG = Retrieval Augmented Generation
    - Retrieval: find relevant chunks from ChromaDB
    - Augmented: add those chunks to the prompt
    - Generation: LLM answers using only those chunks
    """

    def __init__(self, vectorstore: Chroma):
        # Retriever fetches top 5 relevant chunks for any question
        # k=5 means → get 5 most similar chunks
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )

        # LLM that will generate answers
        self.llm = ChatOllama(model="llama3.2")

        # Conversation memory — stores chat history as list
        # Each entry: {"role": "user"/"assistant", "content": "..."}
        self.memory = []

        # Build the RAG chain once during init
        self.chain = self._build_chain()

    def _build_chain(self):
        """
        Builds the RAG chain using LCEL (LangChain Expression Language).
        
        Flow:
        question → retriever → relevant chunks
                             ↓
        prompt = chunks + memory + question → LLM → answer
        """

        # Prompt tells LLM how to behave
        # {context} = retrieved chunks
        # {history} = past conversation
        # {question} = current question
        prompt = ChatPromptTemplate.from_template("""
You are a helpful study assistant. Answer questions using ONLY the context provided.
If the answer is not in the context, say "This information is not in the document."
Keep answers clear and concise.

Conversation History:
{history}

Context from document:
{context}

Current Question: {question}

Answer:""")

        def format_docs(docs):
            # joins all retrieved chunks into one string
            return "\n\n".join(doc.page_content for doc in docs)

        def format_history(history):
            # converts memory list into readable string for prompt
            if not history:
                return "No previous conversation."
            lines = []
            for msg in history:
                role = "You" if msg["role"] == "user" else "AI"
                lines.append(f"{role}: {msg['content']}")
            return "\n".join(lines)

        # LCEL chain — pipes data through each step
        chain = (
            {
                # retrieve relevant chunks for the question
                "context": self.retriever | format_docs,
                # pass history from memory
                "history": lambda x: format_history(self.memory),
                # pass question through unchanged
                "question": RunnablePassthrough()
            }
            | prompt        # fill prompt template
            | self.llm      # send to LLM
            | StrOutputParser()  # extract text from LLM response
        )
        return chain

    def ask(self, question: str) -> str:
        """
        Ask a question and get an answer.
        Automatically saves to memory for context.
        """
        # Get answer from RAG chain
        answer = self.chain.invoke(question)

        # Save to memory AFTER getting answer
        self.memory.append({"role": "user", "content": question})
        self.memory.append({"role": "assistant", "content": answer})

        return answer

    def get_chunks_used(self, question: str) -> list:
        """Returns the actual chunks retrieved for a question."""
        return self.retriever.invoke(question)

    def clear_memory(self):
        """Resets conversation history."""
        self.memory = []
        print("Conversation history cleared.")


# ==========================================
# MAIN TEST SCRIPT
# ==========================================

if __name__ == "__main__":

    # Step 1 — Load knowledge base
    print("Loading knowledge base...")
    vectorstore = process_pdf("sample.txt")
    print("Study Assistant ready!")

    # Step 2 — Create assistant
    assistant = StudyAssistant(vectorstore)

    # ==========================================
    # TEST 1: Basic RAG questions about drones
    # ==========================================
    print("\n" + "="*50)
    print("TESTING RAG WITH: Drone Technology")
    print("="*50)

    questions = [
        "What is a drone?",
        "What are the types of drones?",
        "What are the components of a drone?",
        "What is the role of a flight controller?",
        "What are drone regulations in India?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        answer = assistant.ask(question)
        chunks = assistant.get_chunks_used(question)
        print(f"Answer: {answer}")
        print(f"Chunks used: {len(chunks)}")
        print("-" * 40)

    # ==========================================
    # TEST 2: Conversation memory test
    # ==========================================
    print("\n" + "="*50)
    print("TESTING CONVERSATION MEMORY")
    print("="*50)

    # Clear memory for fresh conversation
    assistant.clear_memory()

    memory_questions = [
        "What industries use drones?",
        "Tell me more about agriculture use.",   # ← needs memory of previous answer
        "What license is needed for that?",      # ← needs memory of drone context
    ]

    for question in memory_questions:
        answer = assistant.ask(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

    # ==========================================
    # TEST 3: Memory structure check
    # ==========================================
    print("\n" + "="*50)
    print(f"Total messages in memory: {len(assistant.memory)}")
    print("Memory structure:")
    for i, msg in enumerate(assistant.memory):
        role = "You" if msg["role"] == "user" else "AI"
        # show only first 60 chars of each message
        print(f"  {i+1}. {role}: {msg['content'][:60]}...")