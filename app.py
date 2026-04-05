import streamlit as st
from src.pdf_processor import process_pdf
from test_rag import StudyAssistant

# ---- How this file works ----
# Streamlit reruns the entire script on every user action
# st.session_state → dictionary that persists between reruns
# Without session_state, everything resets on every click

# ==========================================
# PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 AI Study Assistant")
st.caption("Upload a PDF or TXT file and chat with it!")

# ==========================================
# SESSION STATE SETUP
# ==========================================
# These variables survive between Streamlit reruns

if "assistant" not in st.session_state:
    st.session_state.assistant = None  # StudyAssistant object

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history for display

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False  # tracks if file is loaded

# ==========================================
# FILE UPLOAD SECTION
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your study material",
    type=["pdf", "txt"],  # only allow PDF and TXT
    help="Upload any PDF or TXT file to start chatting"
)

if uploaded_file is not None and not st.session_state.file_processed:
    # Save uploaded file to disk temporarily
    # Streamlit gives us the file in memory — we need it on disk for process_pdf
    file_path = f"temp_{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show spinner while processing
    with st.spinner(f"Processing {uploaded_file.name}... this may take a minute"):
        try:
            vectorstore = process_pdf(file_path)
            st.session_state.assistant = StudyAssistant(vectorstore)
            st.session_state.file_processed = True
            st.session_state.messages = []  # clear old chat
            st.success(f"✅ {uploaded_file.name} loaded! Ask me anything.")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ==========================================
# CHAT SECTION
# ==========================================

# Display all previous messages
# st.chat_message("user") → shows human bubble
# st.chat_message("assistant") → shows AI bubble
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box at bottom of screen
# Only active if a file has been loaded
if st.session_state.file_processed:
    user_input = st.chat_input("Ask a question about your document...")

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # Save user message to display history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Get answer from RAG assistant
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.assistant.ask(user_input)
            st.write(answer)

        # Save assistant answer to display history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    # Show placeholder when no file is uploaded
    st.info("👆 Upload a file above to start chatting!")

# ==========================================
# SIDEBAR — extra info
# ==========================================

with st.sidebar:
    st.header("📊 Session Info")

    if st.session_state.file_processed:
        st.success("File loaded ✅")
        msg_count = len(st.session_state.messages)
        st.metric("Messages in chat", msg_count)

        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.assistant.clear_memory()
            st.rerun()  # refresh the page

        # New file button
        if st.button("📁 Load New File"):
            st.session_state.assistant = None
            st.session_state.messages = []
            st.session_state.file_processed = False
            st.rerun()
    else:
        st.info("No file loaded yet")

    st.divider()
    st.caption("Built with LangChain + ChromaDB + Ollama")